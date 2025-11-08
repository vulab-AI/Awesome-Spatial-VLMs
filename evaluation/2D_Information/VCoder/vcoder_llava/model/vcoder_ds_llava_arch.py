#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .multimodal_adapter.builder import build_seg_projector
from .multimodal_depth_adapter.builder import build_depth_projector

from vcoder_llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, SEG_TOKEN_INDEX, DEPTH_TOKEN_INDEX

class VCoderDSLlavaMetaModel:

    def __init__(self, config):
        super(VCoderDSLlavaMetaModel, self).__init__(config)
        self.config = config

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        
        if hasattr(config, "seg_mm_projector_type"):
            self.seg_mm_projector = build_seg_projector(config)
            
        if hasattr(config, "use_mm2_proj"):
            if config.use_mm2_proj:
                self.mm2_projector = build_vision_projector(config)

        if hasattr(config, "depth_mm_projector_type"):
            self.depth_mm_projector = build_depth_projector(config)
        
        if hasattr(config, "mm_vcoder_lm_emb"):
            self.vcoder_lm_emb = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
            
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_seg_modules(self, model_args, fsdp=None):
        mm_seg_select_layer = model_args.mm_seg_select_layer
        mm_seg_select_feature = model_args.mm_seg_select_feature

        self.config.seg_mm_hidden_size = self.vision_tower.hidden_size
        
        self.config.seg_use_mm_proj = True
        self.config.seg_mm_projector_type = getattr(model_args, 'seg_mm_projector_type', 'linear')
        self.config.mm_seg_select_layer = mm_seg_select_layer
        self.config.mm_seg_select_feature = mm_seg_select_feature

        self.seg_mm_projector = build_seg_projector(self.config)
        self.vcoder_lm_emb = nn.Embedding(self.config.vocab_size, self.config.hidden_size, self.config.pad_token_id)

        # use MLP from pretraining stage
        pretrain_mm2_mlp_adapter = model_args.pretrain_mm2_mlp_adapter
        if getattr(model_args, "use_mm2_proj"):
            self.config.use_mm2_proj = model_args.use_mm2_proj
            self.mm2_projector = build_vision_projector(self.config)

            if pretrain_mm2_mlp_adapter is not None:
                mm2_projector_weights = torch.load(pretrain_mm2_mlp_adapter, map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                self.mm2_projector.load_state_dict(get_w(mm2_projector_weights, 'mm_projector'))  
    
    def initialize_depth_modules(self, model_args, fsdp=None):
        mm_depth_select_layer = model_args.mm_depth_select_layer
        mm_depth_select_feature = model_args.mm_depth_select_feature

        self.config.depth_mm_hidden_size = self.vision_tower.hidden_size
        
        self.config.depth_use_mm_proj = True
        self.config.depth_mm_projector_type = getattr(model_args, 'depth_mm_projector_type', 'linear')
        self.config.mm_depth_select_layer = mm_depth_select_layer
        self.config.mm_depth_select_feature = mm_depth_select_feature

        self.depth_mm_projector = build_depth_projector(self.config)

class VCoderDSLlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_seg_images(self, seg_images):
        seg_features = self.get_model().get_vision_tower()(seg_images)
        seg_features = self.get_model().seg_mm_projector(seg_features)
        return seg_features

    def encode_depth_images(self, depth_images):
        depth_features = self.get_model().get_vision_tower()(depth_images)
        depth_features = self.get_model().seg_mm_projector(depth_features)
        return depth_features
    
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_images_w_seg(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm2_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, seg_images, depth_images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            if seg_images is not None and hasattr(self, 'mm2_projector'):
                image_features = self.encode_images_w_seg(concat_images)
            else:
                image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            if seg_images is not None and hasattr(self, 'mm2_projector'):
                image_features = self.encode_images_w_seg(images)
            else:
                image_features = self.encode_images(images)

        if seg_images is not None:
            if type(seg_images) is list or seg_images.ndim == 5:
                concat_seg_images = torch.cat([image for image in seg_images], dim=0)
                seg_features = self.encode_seg_images(concat_seg_images)
                split_sizes = [image.shape[0] for image in seg_images]
                seg_features = torch.split(seg_features, split_sizes, dim=0)
                seg_features = [x.flatten(0, 1) for x in seg_features]
            else:
                seg_features = self.encode_seg_images(seg_images)
        
        if depth_images is not None:
            is_depth_zero = [torch.mean(d) == 0 for d in depth_images]
            if type(depth_images) is list or depth_images.ndim == 5:
                concat_depth_images = torch.cat([image for image in depth_images], dim=0)
                depth_features = self.encode_depth_images(concat_depth_images)
                split_sizes = [image.shape[0] for image in depth_images]
                depth_features = torch.split(depth_features, split_sizes, dim=0)
                depth_features = [x.flatten(0, 1) for x in depth_features]
            else:
                depth_features = self.encode_depth_images(depth_images)
        else:
            is_depth_zero = [True] * input_ids.shape[0]

        self.get_model().vcoder_lm_emb.weight.data = self.get_model().get_input_embeddings().weight.data.clone()
        
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        cur_seg_idx = 0
        cur_depth_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0 and (cur_input_ids == SEG_TOKEN_INDEX).sum() == 0:
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                cur_image_features = image_features[cur_image_idx]
                half_len = cur_input_ids.shape[0] // 2
                if seg_images is not None:
                    cur_seg_features = seg_features[cur_seg_idx]
                    is_cur_depth_zero = is_depth_zero[cur_depth_idx]
                    if not is_cur_depth_zero:
                        cur_depth_features = depth_features[cur_depth_idx]
                    cur_input_embeds_1 = self.get_model().vcoder_lm_emb(cur_input_ids[:half_len])
                    cur_input_embeds_2 = self.get_model().vcoder_lm_emb(cur_input_ids[half_len:])
                else:
                    cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                    cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                if seg_images is not None:
                    if not is_cur_depth_zero:
                        cur_input_embeds = torch.cat([cur_input_embeds_1, cur_depth_features[0:0], cur_seg_features[0:0], cur_image_features[0:0], cur_input_embeds_2], dim=0)
                    else:
                        cur_input_embeds = torch.cat([cur_input_embeds_1, cur_seg_features[0:0], cur_image_features[0:0], cur_input_embeds_2], dim=0)
                else:
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                cur_seg_idx += 1
                cur_depth_idx += 1
                continue
            
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if seg_images is None:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                else:
                    cur_new_input_embeds.append(self.get_model().vcoder_lm_emb(cur_input_ids[:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            
            if seg_images is not None:
                seg_token_indices = torch.where(cur_input_ids == SEG_TOKEN_INDEX)[0]
                while seg_token_indices.numel() > 0:
                    cur_seg_features = seg_features[cur_seg_idx]
                    seg_token_start = seg_token_indices[0]
                    cur_new_input_embeds.append(cur_seg_features)
                    if labels is not None:
                        cur_new_labels.append(torch.full((cur_seg_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[seg_token_start+1:]
                    cur_seg_idx += 1
                    cur_input_ids = cur_input_ids[seg_token_start+1:]
                    seg_token_indices = torch.where(cur_input_ids == SEG_TOKEN_INDEX)[0]
            
            is_cur_depth_zero = is_depth_zero[cur_depth_idx]
            if not is_cur_depth_zero:
                depth_token_indices = torch.where(cur_input_ids == DEPTH_TOKEN_INDEX)[0]
                while depth_token_indices.numel() > 0:
                    cur_depth_features = depth_features[cur_depth_idx]
                    depth_token_start = depth_token_indices[0]
                    cur_new_input_embeds.append(self.get_model().vcoder_lm_emb(cur_input_ids[:depth_token_start]))
                    cur_new_input_embeds.append(cur_depth_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:depth_token_start])
                        cur_new_labels.append(torch.full((cur_depth_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[depth_token_start+1:]
                    cur_depth_idx += 1
                    cur_input_ids = cur_input_ids[depth_token_start+1:]
                    depth_token_indices = torch.where(cur_input_ids == DEPTH_TOKEN_INDEX)[0]
            else:
                cur_depth_idx += 1
            
            if cur_input_ids.numel() > 0:
                if seg_images is None:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                else:
                    cur_new_input_embeds.append(self.get_model().vcoder_lm_emb(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels