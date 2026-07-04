import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json
import torchvision.transforms.functional as TF
# from dataset import get_data_transforms, MVTecDataset, AnomalyDataset
# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


def get_strong_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop((isize, isize), scale=(0.6, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    return data_transforms


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = sorted(os.listdir(self.img_path))

        for defect_type in defect_types:
            if defect_type == 'good':
                # --- 正常样本 ---
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_paths.sort()
                
                img_tot_paths.extend(img_paths)
                # [修改] 用空字符串 "" 占位，保持全是字符串类型
                gt_tot_paths.extend([""] * len(img_paths)) 
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                # --- 异常样本 ---
                img_dir = os.path.join(self.img_path, defect_type)
                # MVTec: mask 在 ground_truth/<defect_type>/<name>_mask.png
                gt_dir = os.path.join(self.gt_path, defect_type)

                img_paths = glob.glob(img_dir + "/*.png") + \
                            glob.glob(img_dir + "/*.JPG") + \
                            glob.glob(img_dir + "/*.bmp")
                img_paths.sort()
                
                # 查找对应的 Mask (使用我上一条回复的自动配对逻辑)
                valid_img_paths = []
                matched_gt_paths = []
                
                for img_p in img_paths:
                    filename = os.path.basename(img_p)
                    name_no_ext = os.path.splitext(filename)[0]
                    gt_name = name_no_ext + "_mask.png"
                    gt_p = os.path.join(gt_dir, gt_name)
                    
                    if not os.path.exists(gt_p):
                        # 尝试同名文件
                        gt_p = os.path.join(gt_dir, filename)
                    
                    if os.path.exists(gt_p):
                        valid_img_paths.append(img_p)
                        matched_gt_paths.append(gt_p)
                    else:
                        print(f"❌ 找不到 Mask，跳过图片: {img_p}")
                        print(f"   --> 试图寻找的路径是: {gt_p}")
                        pass # 没找到mask就跳过
                        
                img_tot_paths.extend(valid_img_paths)
                gt_tot_paths.extend(matched_gt_paths)
                tot_labels.extend([1] * len(valid_img_paths))
                tot_types.extend([defect_type] * len(valid_img_paths))

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

   
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path

class VisADataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
            self.gt_path = os.path.join(self.img_path, 'ground_truth')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = sorted(os.listdir(self.img_path))

        for defect_type in defect_types:
            if defect_type == 'good':
                # --- 正常样本 ---
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_paths.sort()
                
                img_tot_paths.extend(img_paths)
                # [修改] 用空字符串 "" 占位，保持全是字符串类型
                gt_tot_paths.extend([""] * len(img_paths)) 
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            elif defect_type != 'ground_truth':
                # --- 异常样本 ---
                img_dir = os.path.join(self.img_path, defect_type)
                # VisA: mask 在 ground_truth/<defect_type>/<name>.png
                gt_dir = os.path.join(self.gt_path, defect_type)

                img_paths = glob.glob(img_dir + "/*.png") + \
                            glob.glob(img_dir + "/*.JPG") + \
                            glob.glob(img_dir + "/*.bmp")
                img_paths.sort()
                
                # 查找对应的 Mask (使用我上一条回复的自动配对逻辑)
                valid_img_paths = []
                matched_gt_paths = []
                
                
                for img_p in img_paths:
                    filename = os.path.basename(img_p)
                    name_no_ext = os.path.splitext(filename)[0]
                    # For MVTec
                    gt_name = name_no_ext + ".png"
                    gt_p = os.path.join(gt_dir, gt_name)
                    
                    if not os.path.exists(gt_p):
                        # 尝试同名文件
                        gt_p = os.path.join(gt_dir, filename)
                    
                    if os.path.exists(gt_p):
                        valid_img_paths.append(img_p)
                        matched_gt_paths.append(gt_p)
                    else:
                        print(f"❌ 找不到 Mask，跳过图片: {img_p}")
                        print(f"   --> 试图寻找的路径是: {gt_p}")
                        pass # 没找到mask就跳过
                img_tot_paths.extend(valid_img_paths)
                gt_tot_paths.extend(matched_gt_paths)
                tot_labels.extend([1] * len(valid_img_paths))
                tot_types.extend([defect_type] * len(valid_img_paths))

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

   
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'sup', category + '.json')
        print(json_path)
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # if self.phase == 'train':
        #     return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class MANTADataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'MANTA_TINY_256_cropped', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'sup_cropped', category + '.json')
        print(json_path)
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'MANTA_TINY_256_cropped', category, sample['image_path']))
            label = sample['anomaly_class'] != 'good'
            if label:
                self.gt_paths.append(os.path.join(root, 'MANTA_TINY_256_cropped', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # if self.phase == 'train':
        #     return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class LOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*/000.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        size = (img.size[1], img.size[0])
        img = self.transform(img)
        type = self.types[idx]
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path, type, size


class InsPLADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
        self.transform = transform
        self.phase = phase
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
            else:
                if self.phase == 'train':
                    continue
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)

        return img, label, img_path


class AeBADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        defect_types = [i for i in defect_types if i[0] != '.']
        for defect_type in defect_types:
            if defect_type == 'good':
                domain_types = os.listdir(os.path.join(self.img_path, defect_type))
                domain_types = [i for i in domain_types if i[0] != '.']

                for domain_type in domain_types:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, domain_type) + "/*.png")
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend([0] * len(img_paths))
                    tot_labels.extend([0] * len(img_paths))
                    tot_types.extend(['good'] * len(img_paths))
            else:
                domain_types = os.listdir(os.path.join(self.img_path, defect_type))
                domain_types = [i for i in domain_types if i[0] != '.']

                for domain_type in domain_types:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, domain_type) + "/*.png")
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type, domain_type) + "/*.png")
                    img_paths.sort()
                    gt_paths.sort()
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend(gt_paths)
                    tot_labels.extend([1] * len(img_paths))
                    tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if self.phase == 'train':
            return img, label
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class MiniDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):

        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
            img_tot_paths.extend(img_paths)
            tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path, label = self.img_paths[idx], self.labels[idx]
            img = Image.open(img_path).convert('RGB')
        except:
            img_path, label = self.img_paths[idx - 1], self.labels[idx - 1]
            img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label


class MVTecDRAEMDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, strong_transform, phase, anomaly_source_path, anomaly_ratio=0.5,
                 size=256):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train', 'label')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        self.strong_transform = strong_transform
        self.anomaly_ratio = anomaly_ratio
        self.size = size
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        no_anomaly = random.random()
        if no_anomaly > self.anomaly_ratio:
            return image, 0
        else:
            aug = self.randAugmenter()

            perlin_scale = 6
            min_perlin_scale = 0
            anomaly_source_img = Image.open(anomaly_source_path).convert('RGB').resize((self.size, self.size))
            anomaly_source_img = np.asarray(anomaly_source_img)
            anomaly_img_augmented = aug(image=anomaly_source_img)

            perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

            perlin_noise = rand_perlin_2d_np((self.size, self.size),
                                             (perlin_scalex, perlin_scaley))
            perlin_noise = self.rot(image=perlin_noise)
            threshold = 0.5
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_thr = np.expand_dims(perlin_thr, axis=2)

            img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr

            beta = random.random() * 0.7 + 0.1

            image = image.resize((self.size, self.size))
            image = np.asarray(image)
            augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
            # augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image

            return Image.fromarray(np.uint8(augmented_image)), 1

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')

        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        a_img, label = self.augment_image(img, self.anomaly_source_paths[anomaly_source_idx])

        img = self.transform(img)
        a_img = self.strong_transform(a_img)

        assert img.size()[1:] == a_img.size()[1:], "image.size != a_img.size !!!"

        return img, a_img, label


class MVTecSimplexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform

        self.simplexNoise = Simplex_CLASS()
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img_normal = self.transform(img)

        if random.random() > 0.5:
            return img_normal, img_normal
        ## simplex_noise
        size = 256
        img = img.resize((size, size))
        img = np.asarray(img)
        h_noise = np.random.randint(10, int(size // 8))
        w_noise = np.random.randint(10, int(size // 8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        init_zero = np.zeros((256, 256, 3))
        init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise + w_noise,
        :] = 0.2 * simplex_noise.transpose(1, 2, 0)
        img_noise = img + init_zero * 255
        img_noise = Image.fromarray(np.uint8(img_noise))
        img_noise = self.transform(img_noise)

        return img_normal, img_noise
        
class AnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None):
        """
        Args:
            root_dir (string): 数据集的根目录 (例如 '.../label' 或 '.../unlabel')
            transform (callable, optional): 应用于原图的 transforms (如 Resize, Normalize 等)
            mask_transform (callable, optional): 应用于 Mask 的 transforms (通常只需要 Resize 和 ToTensor)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.image_paths = []
        self.mask_paths = []
        self.labels = [] # 0: good, 1: bad
        
        # 1. 加载 Good (正常样本)
        good_dir = os.path.join(root_dir, 'good')
        if os.path.exists(good_dir):
            for img_name in os.listdir(good_dir):
                if self._is_image_file(img_name):
                    self.image_paths.append(os.path.join(good_dir, img_name))
                    self.mask_paths.append(None) # Good 样本没有物理 mask 文件
                    self.labels.append(0)        # Label 0 代表正常

        # 2. 加载 Bad (异常样本)
        bad_dir = os.path.join(root_dir, 'bad')
        mask_dir = os.path.join(root_dir, 'ground_truth')
        # print(bad_dir)
        if os.path.exists(bad_dir):
            for img_name in os.listdir(bad_dir):
                if self._is_image_file(img_name):
                    img_path = os.path.join(bad_dir, img_name)
                    name, ext = os.path.splitext(img_name)

                    # Candidate mask paths, checked in order. Covers both layouts:
                    #   MVTec : ground_truth/<name>_mask.<ext>  (fallback: ground_truth/<name>.png)
                    #   VisA  : ground_truth/bad/<name>.png    (fallback: ground_truth/bad/<name>_mask.png)
                    gt_bad_dir = os.path.join(mask_dir, 'bad')
                    candidates = [
                        os.path.join(mask_dir, f"{name}_mask{ext}"),
                        os.path.join(mask_dir, f"{name}.png"),
                        os.path.join(gt_bad_dir, f"{name}.png"),
                        os.path.join(gt_bad_dir, f"{name}_mask.png"),
                    ]
                    mask_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.labels.append(1) # Label 1 代表异常
        
    def _is_image_file(self, filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

    def __getitem__(self, idx):
        # 读取原图
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 读取或生成 Mask
        mask_path = self.mask_paths[idx]
        
        if mask_path is not None and os.path.exists(mask_path):
            #如果是 Bad 样本，读取对应的 Mask
            mask = Image.open(mask_path).convert('L') # 转为灰度图
        else:
            # 如果是 Good 样本 (或找不到 mask)，生成全黑 Mask
            # 尺寸与原图一致 (W, H)
            mask = Image.new('L', image.size, color=0)

        label = self.labels[idx]

        # 应用 Transform
        # 注意：如果您使用了 RandomCrop 或 RandomFlip，必须保证 image 和 mask 变换一致
        # 这里展示最基础的分开处理方式
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # 如果没有指定 mask transform，至少要转为 Tensor
            # 建议在这里手动 Resize mask 以匹配 image 的大小 (如果 transform 里有 Resize)
            mask = TF.to_tensor(mask)

        return image, mask, label, img_path 

    def __len__(self):
        return len(self.image_paths)
