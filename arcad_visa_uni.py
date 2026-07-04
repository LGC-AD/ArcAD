import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
import os
import csv
import argparse
import logging
import warnings
from functools import partial
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.init import trunc_normal_

from dataset import VisADataset, AnomalyDataset, get_data_transforms
from models import vit_encoder
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from optimizers import StableAdamW
from utils import evaluation_batch, global_cosine_hm_percent, WarmCosineScheduler, evaluation_fusion

warnings.filterwarnings("ignore")


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        # label 1 gets alpha, label 0 gets (1 - alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Discriminator (SimpleNet-style 2-layer MLP), used as a structural regularizer.
class Discriminator(nn.Module):
    def __init__(self, in_planes, n_layers=2, hidden=None):
        super(Discriminator, self).__init__()
        _hidden = in_planes if hidden is None else hidden
        self.body = nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i + 1),
                                 nn.Sequential(
                                     nn.Linear(_in, _hidden),
                                     nn.BatchNorm1d(_hidden),
                                     nn.LeakyReLU(0.2)
                                 ))
        self.tail = nn.Linear(_hidden, 1, bias=False)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x


class BottleneckPrototypeLearner(nn.Module):
    """SPM: vMF prototype modeling on the hypersphere with Sinkhorn assignment."""
    def __init__(self, feature_dim, num_prototypes=50, num_special_tokens=5,
                 temperature=0.1, momentum=0.99, epsilon=0.05, sinkhorn_iterations=3,
                 noise_std=0.015, cluster_mode='sinkhorn'):
        super().__init__()
        self.num_special_tokens = num_special_tokens
        self.K = num_prototypes
        self.tau = temperature
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.momentum = momentum
        self.noise_std = noise_std
        self.cluster_mode = cluster_mode

        self.register_buffer("prototypes", torch.zeros(num_prototypes, feature_dim))
        self.is_initialized = False

    def normalize_prototypes(self):
        self.prototypes.data = F.normalize(self.prototypes.data, dim=1)

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]

        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            # normalize rows: uniform prototype assignment
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            # normalize cols: each sample assigned to exactly one prototype
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_of_cols
            Q /= B

        Q *= B
        return Q.t()

    def augment_features(self, x):
        """Prototype-restricted synthetic anomaly synthesis (DGC).
        Input:  x [N, C] normalized anchors (real normal features).
        Output: x_aug [N, C] low-likelihood synthetic anomalies, normalized.
        """
        K = 50  # candidates per anchor
        N, C = x.shape
        x_expanded = x.unsqueeze(1).expand(N, K, C)

        # sample v ~ N(z, sigma^2 I) around each anchor
        noise = torch.randn_like(x_expanded)
        x_candidates = x_expanded + self.noise_std * noise
        x_candidates = F.normalize(x_candidates, dim=2)  # re-project onto hypersphere

        # keep the candidate farthest from every prototype (lowest likelihood)
        prototypes = F.normalize(self.prototypes.data, dim=1)            # [M, C]
        candidates_flat = x_candidates.view(N * K, C)
        sim_matrix = torch.mm(candidates_flat, prototypes.t())           # [N*K, M]
        max_sim_per_candidate, _ = sim_matrix.max(dim=1)                 # nearest prototype sim
        max_sim_per_candidate = max_sim_per_candidate.view(N, K)
        _, best_indices = max_sim_per_candidate.min(dim=1)               # lowest nearest-sim

        best_indices = best_indices.view(N, 1, 1).expand(N, 1, C)
        x_aug = torch.gather(x_candidates, 1, best_indices).squeeze(1)
        return x_aug

    def forward(self, x):
        x_spatial = x[:, self.num_special_tokens:, :]
        B, N, C = x_spatial.shape
        x_flat = x_spatial.reshape(-1, C)

        if self.cluster_mode == 'sinkhorn':
            x_norm = F.normalize(x_flat, dim=1)
            p_norm = F.normalize(self.prototypes, dim=1)

            logits = torch.matmul(x_norm, p_norm.t())

            with torch.no_grad():
                if x_flat.shape[0] > self.K:
                    q_weights = self.distributed_sinkhorn(logits)
                else:
                    q_weights = F.softmax(logits / self.epsilon, dim=1)

            # cross-entropy between Sinkhorn target Q and logits
            log_probs = F.log_softmax(logits / self.tau, dim=1)
            loss = -torch.mean(torch.sum(q_weights * log_probs, dim=1))

            return loss, x_norm, q_weights

    @torch.no_grad()
    def update_prototypes(self, x_norm, q_weights):
        """EMA prototype update."""
        z_sum = torch.matmul(q_weights.t(), x_norm)
        count = torch.sum(q_weights, dim=0).unsqueeze(1)  # [K, 1]

        if self.cluster_mode == 'sinkhorn':
            z_sum = F.normalize(z_sum, dim=1)
            self.prototypes.data = self.prototypes.data * self.momentum + z_sum * (1 - self.momentum)

        self.prototypes.data = F.normalize(self.prototypes.data, dim=1)

    def calculate_contrastive_calibration_loss(self, x, mask, z_syn):
        """DGC contrastive calibration: push real anomalies away from normal
        prototypes, pull them toward synthetic anomalies."""
        # 1. real anomaly features as anchors
        x_spatial = x[:, self.num_special_tokens:, :]
        B, N, C = x_spatial.shape
        H = int(math.sqrt(N))

        if mask.shape[-1] != H:
            mask_down = F.adaptive_max_pool2d(mask, (H, H))
        else:
            mask_down = mask

        mask_flat = mask_down.reshape(B, -1)
        x_all = x_spatial.reshape(-1, C)
        mask_all = mask_flat.reshape(-1)
        anchor_feats = x_all[mask_all > 0]

        if anchor_feats.shape[0] == 0:
            return torch.tensor(0.0).to(x.device)

        anchor_norm = F.normalize(anchor_feats, dim=1)

        # 2. push loss: drive anchors off the normal boundary (nearest prototype)
        p_norm = F.normalize(self.prototypes, dim=1)
        sim_matrix_push = torch.matmul(anchor_norm, p_norm.t())
        max_sim_push, _ = torch.max(sim_matrix_push, dim=1)
        loss_push = F.relu(max_sim_push).mean()

        # 3. pull loss: attract anchors toward synthetic anomalies
        if z_syn is None or z_syn.shape[0] == 0:
            loss_pull = torch.tensor(0.0).to(x.device)
        else:
            z_syn_norm = F.normalize(z_syn, dim=1)
            sim_matrix_pull = torch.matmul(anchor_norm, z_syn_norm.t())
            avg_sim_pull = torch.mean(sim_matrix_pull, dim=1)
            loss_pull = (1.0 - avg_sim_pull).mean()

        return loss_push + loss_pull


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)
    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train(item_list):
    setup_seed(1)

    total_iters = 10000
    batch_size = 16
    image_size = 448
    crop_size = 392

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list_label = []
    test_data_list = []

    for i, item in enumerate(item_list):
        root_path = args.data_path
        train_path = os.path.join(args.data_path, item, 'train')
        test_path = os.path.join(args.data_path, item)
        label_path = os.path.join(train_path)
        # label convention: 0 = normal, 1 = anomaly
        train_data_label = AnomalyDataset(root_dir=label_path, transform=data_transform, mask_transform=gt_transform)
        test_data = VisADataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")

        train_data_list_label.append(train_data_label)
        test_data_list.append(test_data)

    print_fn(f"Concatenating {len(train_data_list_label)} datasets...")
    combined_train_dataset = ConcatDataset(train_data_list_label)

    train_dataloader_label = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    encoder_name = 'dinov2reg_vit_base_14'

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    bottleneck = []
    decoder = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.4))
    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.,
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)

    # learner + discriminator
    num_special_tokens = 5

    learner = BottleneckPrototypeLearner(
        feature_dim=embed_dim,
        num_prototypes=500,
        num_special_tokens=num_special_tokens,
        noise_std=0.015,
        cluster_mode=args.cluster_mode
    ).to(device)
    discriminator = Discriminator(in_planes=embed_dim).to(device)

    # only bottleneck / decoder / discriminator are trained (encoder frozen)
    trainable = nn.ModuleList([bottleneck, decoder, discriminator])

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)

    # alpha=0.75 with flipped labels keeps the original per-class weights:
    # normal(target 0) -> 1-0.75=0.25, anomaly/synthetic(target 1) -> 0.75
    criterion_cls = BinaryFocalLoss(alpha=0.75, gamma=2.0).to(device)

    proto_init_path = args.proto_path
    if os.path.exists(proto_init_path):
        print_fn(f"Found pre-calculated prototypes at {proto_init_path}. Loading...")
        loaded_protos = torch.load(proto_init_path, map_location=device)
        learner.prototypes.data.copy_(loaded_protos)
        learner.is_initialized = True
        print_fn("Prototypes loaded successfully!")
    else:
        print_fn("Pre-calculated prototypes NOT found. Fallback to online K-Means initialization (Slow & Small sample)...")

    print_fn("Initialization done. Starting training loop...")

    # loss weights
    LAMBDA_COMP = 0.1   # Sinkhorn clustering
    LAMBDA_CLS = 0.1    # discriminator
    LAMBDA_PUSH = 0.1   # DGC contrastive calibration

    print_fn(f'Start training... Images: {len(combined_train_dataset)}')
    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader_label)))):
        model.train()
        discriminator.train()

        loss_list = []
        for img, mask, label, img_path in train_dataloader_label:
            img = img.to(device)
            mask = mask.to(device)
            label = label.to(device)

            en, de, blk = model(img)  # blk: [B, Tokens, C]

            loss_recon = torch.tensor(0.0).to(device)
            loss_comp = torch.tensor(0.0).to(device)
            loss_cls = torch.tensor(0.0).to(device)
            loss_push = torch.tensor(0.0).to(device)

            normal_indices = (label == 0).nonzero(as_tuple=True)[0]
            anom_indices = (label == 1).nonzero(as_tuple=True)[0]

            current_batch_z_syn = None

            # --- Part 1: normal samples (recon + clustering + discriminator) ---
            if len(normal_indices) > 0:
                en_norm = [e[normal_indices] for e in en]
                de_norm = [d[normal_indices] for d in de]

                p = min(0.9 * it / 1000, 0.9)
                loss_recon = global_cosine_hm_percent(en_norm, de_norm, p=p, factor=0.1)

                # Sinkhorn clustering loss + prototype EMA update
                blk_norm = blk[normal_indices]
                loss_comp, x_feats_norm, q_weights = learner(blk_norm)
                learner.update_prototypes(x_feats_norm, q_weights)

                # discriminator: normal features -> target 0
                scores_real = discriminator(x_feats_norm)
                loss_real = criterion_cls(scores_real, torch.zeros_like(scores_real))

                # synthesize pseudo-anomalies, keep as this batch's synthetic pool
                fake_anom_feats = learner.augment_features(x_feats_norm)
                current_batch_z_syn = fake_anom_feats.detach()
                scores_fake = discriminator(fake_anom_feats)
                loss_fake = criterion_cls(scores_fake, torch.ones_like(scores_fake))

                loss_cls = loss_cls + (loss_real + loss_fake) * 0.5

            # --- Part 2: real anomalies (discriminator) + DGC calibration ---
            if len(anom_indices) > 0:
                blk_anom = blk[anom_indices]
                blk_spatial_anom = blk_anom[:, num_special_tokens:, :]

                B_a, N_a, C_a = blk_spatial_anom.shape
                x_flat_anom = blk_spatial_anom.reshape(-1, C_a)
                x_flat_anom = F.normalize(x_flat_anom, dim=1)

                # discriminator: real anomaly features -> target 1
                scores_anom_real = discriminator(x_flat_anom)
                loss_anom_real = criterion_cls(scores_anom_real, torch.ones_like(scores_anom_real))
                loss_cls = loss_cls + loss_anom_real

                blk_anom = blk[anom_indices]
                mask_anom = mask[anom_indices]

                # DGC contrastive calibration
                if current_batch_z_syn is None:
                    temp_protos = F.normalize(learner.prototypes, dim=1)
                    current_batch_z_syn = learner.augment_features(temp_protos).detach()
                    loss_push = 0

                loss_push = learner.calculate_contrastive_calibration_loss(blk_anom, mask_anom, current_batch_z_syn)

            loss = loss_recon + LAMBDA_COMP * loss_comp + LAMBDA_CLS * loss_cls + LAMBDA_PUSH * loss_push

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()

            loss_list.append(loss.item())
            lr_scheduler.step()

            if (it + 1) % total_iters == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

                csv_rows = []
                csv_header = ['Category', 'I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']
                save_path_dir = os.path.join(args.save_dir, args.save_name)
                if not os.path.exists(save_path_dir):
                    os.makedirs(save_path_dir)

                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                  num_workers=4)

                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
                    csv_rows.append([item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px])

                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

                mean_metrics = [
                    np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                    np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)
                ]
                csv_rows.append(['Mean'] + mean_metrics)

                csv_file_path = os.path.join(save_path_dir, 'results.csv')
                try:
                    with open(csv_file_path, mode='w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(csv_header)
                        writer.writerows(csv_rows)
                    print_fn(f"Results successfully saved to {csv_file_path}")
                except Exception as e:
                    print_fn(f"Error saving CSV: {e}")

                model.train()
                discriminator.train()

            it += 1
            if it == total_iters:
                break
            if (it + 1) % 100 == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []

    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/home/hanningning/AD/VisA_CD')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='vitill_uni_simplesnet_discriminator_focal')
    parser.add_argument('--cluster_mode', type=str, default='sinkhorn')
    parser.add_argument('--proto_path', type=str, default='./VisA_S3_init_files500/prototypes_init.pth',
                        )
    args = parser.parse_args()

    item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    train(item_list)
