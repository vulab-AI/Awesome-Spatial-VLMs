import random
import os
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed as dist
import torch
import math
from mmdet.datasets import DATASETS
import cv2
from .metrics import SiLogLoss, cropping_img, eval_depth, display_result


class BaseDataset(Dataset):
    metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
                   'log10', 'silog']

    def __init__(self, crop_size):

        self.count = 0

        basic_transform = [
            A.HorizontalFlip(),
            A.RandomCrop(crop_size[0], crop_size[1]),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()
        ]
        self.basic_transform = basic_transform
        self.to_tensor = transforms.ToTensor()

        self.criterion_val = SiLogLoss()

    def readTXT(self, txt_path):
        with open(txt_path, 'r') as f:
            listInTXT = [line.strip() for line in f]

        return listInTXT

    def augment_training_data(self, image, depth):
        H, W, C = image.shape

        if self.count % 4 == 0:
            alpha = random.random()
            beta = random.random()
            p = 0.75

            l = int(alpha * W)
            w = int(max((W - alpha * W) * beta * p, 1))

            image[:, l:l+w, 0] = depth[:, l:l+w]
            image[:, l:l+w, 1] = depth[:, l:l+w]
            image[:, l:l+w, 2] = depth[:, l:l+w]

        additional_targets = {'depth': 'mask'}
        aug = A.Compose(transforms=self.basic_transform,
                        additional_targets=additional_targets)
        augmented = aug(image=image, depth=depth)
        image = augmented['image']
        depth = augmented['depth']

        image = self.to_tensor(image)
        depth = self.to_tensor(depth)

        self.count += 1

        return image, depth

    def augment_test_data(self, image, depth):
        image = self.to_tensor(image)
        depth = self.to_tensor(depth)

        return image, depth


@DATASETS.register_module()
class nyudepthv2(BaseDataset):
    def __init__(self, data_path, filenames_path='./task/depth/filenames/', train_file='/train_list.txt',
                 is_train=True, crop_size=(448, 576), scale_size=None, crop_boundary=False):
        super().__init__(crop_size)
        if crop_boundary:
            basic_transform = [
                A.Crop(x_min=41, y_min=0, x_max=601, y_max=480),
                A.HorizontalFlip(),
                A.RandomCrop(crop_size[0], crop_size[1]),
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
                A.HueSaturationValue()
            ]
        else:
            basic_transform = [
                A.HorizontalFlip(),
                A.RandomCrop(crop_size[0], crop_size[1]),
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
                A.HueSaturationValue()
            ]
        self.basic_transform = basic_transform

        self.scale_size = scale_size

        self.is_train = is_train
        self.data_path = os.path.join(data_path, 'nyu_depth_v2')

        self.image_path_list = []
        self.depth_path_list = []

        txt_path = os.path.join(filenames_path, 'nyudepthv2')
        if is_train:
            txt_path += train_file
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
        filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)

        depth = depth / 1000.0  # convert in meters

        return {'img': image, 'depth': depth, 'filename': [filename], 'task_type': 'depth'}

    def evaluate(self, results, logger=None, **eval_kwargs):
        result_metrics = {}
        for metric in self.metric_name:
            result_metrics[metric] = 0.0

        pred_ds, depth_gts = zip(*results)
        pred_ds, depth_gts = torch.stack(pred_ds), torch.stack(depth_gts)

        total_size = len(results)
        assert total_size == len(self.filenames_list)

        for pred_d, depth_gt in results:
            pred_crop, gt_crop = cropping_img(pred_d, depth_gt)
            computed_result = eval_depth(pred_crop, gt_crop)

            for key in self.metric_name:
                result_metrics[key] += computed_result[key]

        for key in self.metric_name:
            result_metrics[key] = result_metrics[key] / total_size

        logger.info(display_result(result_metrics))

        lst = []
        for metric, value in result_metrics.items():
            lst.append(f'{value:.4f}')
        lst = ' '.join(lst)
        result_metrics['depth_copypaste'] = lst

        return result_metrics
