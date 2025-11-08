import os
import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
from torchvision import transforms as T
from torch.utils.data import Dataset


def MaskGenerator(depth, mask_ratio=0.10, mask_patch_size=16):
    B, H, W = depth.shape
    assert H % mask_patch_size == 0 and W % mask_patch_size == 0 and mask_ratio > 0.0

    rand_size = (H // mask_patch_size) * (W // mask_patch_size)
    mask_count = int(np.ceil(rand_size * mask_ratio))

    mask_idx = np.random.permutation(rand_size)[:mask_count]
    mask = np.zeros(rand_size, dtype=float)
    mask[mask_idx] = 1.0

    mask = 1 - mask.reshape(B, (H // mask_patch_size), (W // mask_patch_size))
    mask = torch.from_numpy(mask.repeat(
        mask_patch_size, axis=1).repeat(mask_patch_size, axis=2))
    mask_depth = mask * depth
    return mask_depth


class NYUDepthV2(Dataset):
    def __init__(self, data_path, filenames_path='./dataset/filenames/',
                 is_train=False, crop_size=(480, 480), scale_size=None,
                 mask=False, mask_ratio=0.10, mask_patch_size=16):
        self.scale_size = scale_size
        self.is_train = is_train
        self.data_path = data_path
        self.image_path_list = []
        self.depth_path_list = []
        self.write_flag = 1
        self.mask = mask
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        transform = [
            A.Crop(x_min=41, y_min=0, x_max=601, y_max=480),
            A.HorizontalFlip(),
            A.RandomCrop(crop_size[0], crop_size[1]),
        ]
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        txt_path = filenames_path
        if is_train:
            txt_path += '/train_list.txt'
        else:
            txt_path += '/test_list.txt'
            self.data_path = self.data_path + '/official_splits/test/'

        self.filenames_list = self.readTXT(txt_path)
        phase = 'train' if is_train else 'test'
        print("Dataset: NYU Depth V2")
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path = self.data_path + self.filenames_list[idx].split(' ')[0]
        gt_path = self.data_path + self.filenames_list[idx].split(' ')[1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))

        if self.is_train:
            depth = self.augment_depth_train_data(depth)
            depth = depth / 1000.0
            if self.mask:
                mask_depth = MaskGenerator(
                    depth, self.mask_ratio, self.mask_patch_size)
                return depth, mask_depth
            else:
                return depth, depth
        else:
            depth = self.augment_depth_test_data(depth)
            depth = depth / 1000.0
            return depth, depth

    def readTXT(self, txt_path):
        with open(txt_path, 'r') as f:
            listInTXT = [line.strip() for line in f]
        return listInTXT

    def augment_depth_train_data(self, depth):
        # additional_targets = {'depth': 'mask'}
        aug = A.Compose(transforms=self.transform,
                        # additional_targets=additional_targets
                        )
        augmented = aug(image=depth)
        depth = augmented['image']
        depth = self.to_tensor(depth).squeeze().unsqueeze(dim=0)
        return depth

    def augment_depth_test_data(self, depth):
        depth = self.to_tensor(depth).squeeze().unsqueeze(dim=0)
        return depth
