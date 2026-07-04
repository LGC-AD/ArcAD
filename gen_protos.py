import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, ConcatDataset
from functools import partial

from dataset import (
    RealIADDataset, MANTADataset, MVTecDataset, VisADataset,
    AnomalyDataset, get_data_transforms,
)
from models import vit_encoder
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2


def get_feature_extractor(device, embed_dim=768, num_heads=12):
    encoder_name = 'dinov2reg_vit_base_14'
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    print(f"Loading Encoder: {encoder_name}...")
    encoder = vit_encoder.load(encoder_name)

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.4)])
    decoder = nn.ModuleList([VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                                      attn_drop=0., attn=LinearAttention2) for _ in range(8)])

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    return model.to(device)


def process_and_save_cluster(model, dataset, device, args, save_filename):
    """Extract bottleneck features from normal samples, cluster, and save prototypes."""
    print(f"\nProcessing target: {save_filename}")
    print(f"Total labeled images: {len(dataset)}")

    BATCH_SIZE = 32
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    features_buffer = []
    TARGET_PATCHES = 1000000  # early-stop cap

    print(f"Extracting features...")
    start_time = time.time()
    normal_img_count = 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if len(batch) == 4:
                img, mask, label, _ = batch
            elif len(batch) == 3:
                img, _, label = batch
            else:
                img, label = batch[0], batch[1]

            img = img.to(device)
            label = label.to(device)

            normal_mask = (label == 0)
            if not normal_mask.any():
                continue

            img = img[normal_mask]
            normal_img_count += img.shape[0]

            _, _, blk = model(img)
            blk_spatial = blk[:, 5:, :]            # drop special tokens
            feats_flat = blk_spatial.reshape(-1, blk_spatial.shape[-1])

            num_keep = int(feats_flat.shape[0] * 0.3)   # keep 30% of patches
            if num_keep > 0:
                idx = torch.randperm(feats_flat.shape[0])[:num_keep]
                feats_sampled = feats_flat[idx]
                feats_norm = F.normalize(feats_sampled, dim=1).cpu().numpy()
                features_buffer.append(feats_norm)

            current_count = sum(f.shape[0] for f in features_buffer)
            if i % 20 == 0:
                print(f"  Batch {i}: collected {current_count} patches...")
            if current_count >= TARGET_PATCHES:
                break

    extract_time = time.time() - start_time
    print(f"Feature extraction finished in {extract_time:.2f}s.")

    if len(features_buffer) == 0:
        print(f"Warning: No features collected for {save_filename}! Skipping...")
        return

    all_feats = np.concatenate(features_buffer, axis=0)
    if all_feats.shape[0] > TARGET_PATCHES:
        all_feats = all_feats[:TARGET_PATCHES]

    print(f"Final feature shape for Clustering: {all_feats.shape}")

    print(f"Running MiniBatchKMeans (K={args.num_prototypes})...")
    cluster_start_time = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=args.num_prototypes,
        batch_size=16384,
        n_init=1,
        max_no_improvement=20,
        random_state=42,
        verbose=0,
    )
    kmeans.fit(all_feats)
    print(f"Clustering finished in {time.time() - cluster_start_time:.2f}s.")

    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
    centers = F.normalize(centers, dim=1)

    os.makedirs(args.save_dir, exist_ok=True)
    full_save_path = os.path.join(args.save_dir, save_filename)
    torch.save(centers, full_save_path)
    print(f"Prototypes saved to: {full_save_path}")
    print("-" * 50)


def build_datasets(args, data_transform, gt_transform):
    """Build the per-class labeled dataset list for the selected dataset.

    Driven by --dataset when set; otherwise falls back to the active
    commented block below (manual comment-switch mode)."""
    datasets = []

    if args.dataset == 'manta':
        for item in args.item_list:
            datasets.append(MANTADataset(root=args.data_path, category=item,
                                         transform=data_transform, gt_transform=gt_transform, phase='labeled'))
        return datasets

    if args.dataset == 'realiad':
        for item in args.item_list:
            datasets.append(RealIADDataset(root=args.data_path, category=item,
                                           transform=data_transform, gt_transform=gt_transform, phase='labeled'))
        return datasets

    if args.dataset == 'mvtec':
        for item in args.item_list:
            label_dir = os.path.join(args.data_path, item, 'train', 'label')
            datasets.append(AnomalyDataset(root_dir=label_dir, transform=data_transform, mask_transform=gt_transform))
        return datasets

    if args.dataset == 'visa':
        for item in args.item_list:
            label_dir = os.path.join(args.data_path, item, 'train')
            datasets.append(AnomalyDataset(root_dir=label_dir, transform=data_transform, mask_transform=gt_transform))
        return datasets

    # ---- Manual comment-switch fallback (uncomment exactly one block) ----
    # for item in args.item_list:                                   # MANTA
    #     datasets.append(MANTADataset(root=args.data_path, category=item,
    #                                  transform=data_transform, gt_transform=gt_transform, phase='labeled'))
    # for item in args.item_list:                                   # MVTec (cold-start)
    #     label_dir = os.path.join(args.data_path, item, 'train', 'label')
    #     datasets.append(AnomalyDataset(root_dir=label_dir, transform=data_transform, mask_transform=gt_transform))
    for item in args.item_list:                                    # VisA (cold-start)
        label_dir = os.path.join(args.data_path, item, 'train')
        datasets.append(AnomalyDataset(root_dir=label_dir, transform=data_transform, mask_transform=gt_transform))
    return datasets


DATASET_CONFIG = {
    'manta':   dict(data_path='/home/hanningning/AD/MANTA_CD',      save_dir='./MANTA_init_files800',     num_prototypes=800),
    'mvtec':   dict(data_path='/home/hanningning/AD/mvtec_CD',      save_dir='./MVTec_init_files800',    num_prototypes=800),
    'visa':    dict(data_path='/home/hanningning/AD/VisA_CD',       save_dir='./VisA_S3_init_files500',  num_prototypes=500),
    'realiad': dict(data_path='/home/hanningning/AD/Real-IAD_CD',   save_dir='./RealIAD_init_files500', num_prototypes=500),
}

ITEM_LISTS = {
    'mvtec': ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
              'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper'],
    'visa': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
             'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
    'manta': [
        'block_inductor', 'button', 'capsule', 'coated_tablet', 'coffee_beans',
        'copper_standoff', 'embossed_tablet', 'flat_nut', 'gear', 'goji_berries',
        'led', 'led_pad', 'lettered_tablet', 'long_button', 'maize',
        'nut', 'nut_cap', 'oblong_tablet', 'paddy', 'pink_tablet',
        'pistachios', 'power_inductor', 'red_tablet', 'red_washer', 'round_button_cap',
        'screw', 'short_button', 'soybean', 'square_button_cap', 'terminal',
        'thin_resistor', 'type_c', 'wafer_resistor', 'wheat', 'white_tablet',
        'wire_cap', 'yellow_green_washer', 'yellow_tablet',
    ],
    'realiad': ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper'],
}


def run(args):
    device = os.environ.get('GEN_DEV', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = get_feature_extractor(device)
    data_transform, gt_transform = get_data_transforms(448, 392)

    datasets = build_datasets(args, data_transform, gt_transform)

    if args.separate_classes:
        print(">>> Mode: Separate Classes (one prototype file per class)")
        for item, ds in zip(args.item_list, datasets):
            print(f"\nCurrently processing class: {item}")
            process_and_save_cluster(model, ds, device, args, f'prototypes_init_{item}.pth')
    else:
        print(">>> Mode: Concatenated (one global prototype file)")
        combined_dataset = ConcatDataset(datasets)
        process_and_save_cluster(model, combined_dataset, device, args, 'prototypes_init.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # --dataset selects everything (paths, K, item list, dataset class) at once.
    # Leave unset to use the manual comment-switch fallback in build_datasets().
    parser.add_argument('--dataset', type=str, default=None, choices=['mvtec', 'visa', 'manta', 'realiad'])
    parser.add_argument('--data_path', type=str, default='/home/hanningning/AD/VisA_CD')
    parser.add_argument('--save_dir', type=str, default='./VisA_S3_init_files500')
    parser.add_argument('--num_prototypes', type=int, default=500)

    # Concatenated mode (default) -> single prototypes_init.pth used by arcad training.
    # Add --separate_classes to emit one file per class instead.
    parser.add_argument('--separate_classes', action='store_true',
                        help='If true, generate prototypes for each class separately.')

    args = parser.parse_args()

    if args.dataset is not None:
        cfg = DATASET_CONFIG[args.dataset]
        args.data_path = cfg['data_path']
        args.save_dir = cfg['save_dir']
        args.num_prototypes = cfg['num_prototypes']
        args.item_list = ITEM_LISTS[args.dataset]

    run(args)
