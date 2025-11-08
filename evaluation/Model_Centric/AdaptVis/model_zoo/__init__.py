import os
import torch
import clip
from PIL import Image
from torchvision import transforms



def get_model(model_name, device, method='base',root_dir='data'):
    """
    Helper function that returns a model and a potential image preprocessing function.
    """
    if "openai-clip" in model_name:
        from .clip_models import CLIPWrapper
        variant = model_name.split(":")[1]
        model, image_preprocess = clip.load(variant, device=device, download_root=root_dir)
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess

    elif 'spatial_ft' in model_name:
        from .clip_models import CLIPWrapper
        variant = 'ViT-B/32'
        model, image_preprocess = clip.load(variant, device=device, download_root=root_dir)
        print('Getting model weights from {}'.format('data/{}.pt'.format(model_name)))
        state = torch.load('data/{}.pt'.format(model_name))
        state['model_state_dict'] = {k.replace('module.clip_model.', '') : v for k, v in state['model_state_dict'].items()}
        model.load_state_dict(state['model_state_dict'])
        model = model.eval()
        clip_model = CLIPWrapper(model, device)
        return clip_model, image_preprocess


    if model_name == "llava1.5":
        from .llava15 import LlavaWrapper
        llava_model = LlavaWrapper(root_dir=root_dir, device=device,method=method)
        image_preprocess = None
        return llava_model, image_preprocess
    
    elif model_name == "llava1.6":
        from .llava16 import LlavaWrapper
        llava_model = LlavaWrapper(root_dir=root_dir, device=device,method=method)
        image_preprocess = None
        return llava_model, image_preprocess
   
   
        
    else:
        raise ValueError(f"Unknown model {model_name}")
