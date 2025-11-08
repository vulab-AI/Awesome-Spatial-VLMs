import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.datasets import ImageFolder


class CustomToTensor:
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        pic = np.array(pic).astype('float32')
        return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ClassRemap:
    def __call__(self, x):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        x -= 1
        return x


class Uint8Remap:
    def __call__(self, x):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        x /= 255
        return x


def image_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.copy()
        # return img


def save_model(path, vae):
    save_obj = {
        # 'hparams': vae.hparams,
        # vae_params,
        'weights': vae.state_dict()
    }

    torch.save(save_obj, path)


def build_data_set(data):
    pipline_list = []

    transform_dict = {
        'Resize': T.Resize,
        'CenterCrop': T.CenterCrop,
        'CustomToTensor': CustomToTensor,
        'ClassRemap': ClassRemap,
        'Uint8Remap': Uint8Remap,
    }

    for t in data["pipeline"]:
        type = t['type']
        args = t.copy()
        del args['type']
        pipline_list.append(transform_dict[type](**args))

    pipline_list = T.Compose(pipline_list)
    ds = ImageFolder(
        data["image_folder"],
        pipline_list,
        loader=image_loader
    )
    return ds
