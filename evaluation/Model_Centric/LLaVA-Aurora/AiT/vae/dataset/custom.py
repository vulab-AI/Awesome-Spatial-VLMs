import os
import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
from torchvision import transforms as T
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import PIL
import os
import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
from torchvision import transforms as T
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, data_path,
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
        # print(crop_size)
        transform = [
            T.Resize(crop_size,  interpolation = InterpolationMode.NEAREST),
            T.ToTensor(),
            T.Normalize((0.5), (0.5))
        ]
        self.transform = transform

        self.test_transform = [
            T.Resize(crop_size,  interpolation = InterpolationMode.NEAREST),
            T.ToTensor(),
            T.Normalize((0.5), (0.5))
        ]

        self.filenames_list = os.listdir(data_path)
        phase = 'train' if is_train else 'test'
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        gt_path = self.data_path + self.filenames_list[idx]
        
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)[:,:, 0]
  
        depth = Image.fromarray(depth)
    
        if self.is_train:
            depth = self.augment_depth_train_data(depth)
            return depth, depth
        else:
           
            depth = self.augment_depth_test_data(depth)
          
            return depth, depth, self.filenames_list[idx]

   

    def augment_depth_train_data(self, depth):
        depth = T.Compose(transforms=self.transform)(depth)
        depth = depth.squeeze().unsqueeze(dim=0)
        return depth

    def augment_depth_test_data(self, depth):
        depth = T.Compose(transforms=self.test_transform)(depth)
    
        depth = depth.squeeze().unsqueeze(dim=0)
        return depth
