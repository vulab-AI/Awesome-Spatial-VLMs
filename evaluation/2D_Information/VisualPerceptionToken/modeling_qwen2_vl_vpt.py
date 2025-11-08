import os
import math
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModel, SamModel, SamProcessor
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.cache_utils import Cache, StaticCache

from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLCausalLMOutputWithPast, Qwen2VLModel

class Projector(nn.Module):
    """
    A neural network module that performs cosine selection using linear layers and layer normalization.
    Args:
        input_dim (int): The dimension of the input features.
        output_dim (int): The dimension of the output features.
        projector_scale (int): The scale factor for the projector.
    Attributes:
        projector_scale (int): The scale factor for the projector.
        input_dim (int): The dimension of the input features.
        output_dim (int): The dimension of the output features.
        linear_feature (nn.Linear): Linear layer for transforming input features.
        linear_condition (nn.Linear): Linear layer for transforming condition features.
        layernorm_feature (nn.LayerNorm): Layer normalization for input features.
        layernorm_condition (nn.LayerNorm): Layer normalization for condition features.
    Methods:
        reset_parameters():
            Initializes the parameters of the linear layers and layer normalization layers.
        forward(hidden_states, conditions=None):
            Forward pass of the projector.
            Args:
                hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
                conditions (torch.Tensor, optional): Condition tensor of shape (batch_size, output_dim). Default is None.
            Returns:
                torch.Tensor: Transformed tensor of shape (batch_size * seq_len, output_dim).
    """
    # a projector performs cosine selection
    def __init__(self, input_dim, output_dim, projector_scale):
        super().__init__()
        self.projector_scale = projector_scale
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_feature = nn.Linear(input_dim, output_dim)
        self.linear_condition = nn.Linear(output_dim * self.projector_scale, output_dim)
        # layernorms
        self.layernorm_feature = nn.LayerNorm(input_dim)
        self.layernorm_condition = nn.LayerNorm(output_dim * self.projector_scale)

        self.reset_parameters()

    def reset_parameters(self):

        if self.linear_feature.weight.shape[0] == 0:
            self.linear_feature = nn.Linear(self.input_dim, self.output_dim)
        if self.linear_condition.weight.shape[0] == 0:    
            self.linear_condition = nn.Linear(self.output_dim, self.output_dim)
        if self.layernorm_feature.weight.shape[0] == 0:
            self.layernorm_feature = nn.LayerNorm(self.input_dim)
        if self.layernorm_condition.weight.shape[0] == 0:
            self.layernorm_condition = nn.LayerNorm(self.output_dim)

        nn.init.xavier_uniform_(self.linear_feature.weight)
        nn.init.zeros_(self.linear_feature.bias)

        nn.init.xavier_uniform_(self.linear_condition.weight)
        nn.init.zeros_(self.linear_condition.bias)

        nn.init.ones_(self.layernorm_feature.weight)
        nn.init.zeros_(self.layernorm_feature.bias)

        nn.init.ones_(self.layernorm_condition.weight)
        nn.init.zeros_(self.layernorm_condition.bias)

    def forward(self, hidden_states, conditions = None):
        # hidden_states: (batch_size, seq_len, input_dim)
        # conditions: (batch_size, output_dim)

        hidden_states = self.layernorm_feature(hidden_states)
        hidden_states = self.linear_feature(hidden_states)  # (batch_size, seq_len, output_dim)

        if conditions is not None:
            conditions = conditions.view(-1, self.projector_scale * conditions.shape[-1])
            conditions = self.layernorm_condition(conditions)
            conditions = self.linear_condition(conditions)  # (batch_size, output_dim)
            attention = torch.einsum("bso,bo->bs", hidden_states, conditions).contiguous()
            hidden_states = torch.einsum("bs,bso->bso", attention, hidden_states).contiguous()  # (batch_size, seq_len, output_dim)
        
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        return hidden_states

class MinimalProjector(Projector):
    """
    A projector that performs minimal cross attention when the visual encoder is DINO.
    Args:
        input_dim (int): The dimension of the input features.
        output_dim (int): The dimension of the output features.
        projector_scale (int): The scale factor for the projector.
    Methods:
        forward(hidden_states, conditions=None, output_intermediate=False):
            Performs the forward pass of the projector.
            Args:
                hidden_states (torch.Tensor): The input hidden states with shape (batch_size, seq_len, input_dim).
                conditions (torch.Tensor, optional): The conditions with shape (batch_size, output_dim). Defaults to None.
                output_intermediate (bool, optional): Whether to output intermediate results. Defaults to False.
            Returns:
                Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]: The output tensor with shape (batch_size * seq_len, output_dim) and 
                optionally a dictionary of intermediate outputs containing attention, value, and hidden_states if output_intermediate is True.
    """
    # a projector performs minimal cross attention
    def __init__(self, input_dim, output_dim, projector_scale):
        super().__init__(input_dim, output_dim, projector_scale)

    def forward(self, hidden_states, conditions = None, output_intermediate = False):
        # hidden_states: (batch_size, seq_len, input_dim)
        # conditions: (batch_size, output_dim)

        hidden_states = self.layernorm_feature(hidden_states)
        hidden_states = self.linear_feature(hidden_states)  # (batch_size, seq_len, output_dim)

        if conditions is not None:
            conditions = conditions.view(-1, self.projector_scale * conditions.shape[-1])
            conditions = self.layernorm_condition(conditions)
            conditions = self.linear_condition(conditions)  # (batch_size, output_dim)
            
            conditions_key = conditions # (batch_size, output_dim)
            attention = torch.einsum("bso,bo->bs", hidden_states, conditions_key).contiguous()

            conditions_value = conditions  # (batch_size, output_dim)

            value = torch.einsum("bs,bo-> bso", attention, conditions_value).contiguous()
            output =  value + hidden_states
        else:
            output = hidden_states
        
        output = output.view(-1, output.shape[-1])

        if output_intermediate and conditions is not None:
            intermediate_outputs = {
                "attention": attention,
                "value": value,
                "hidden_states": hidden_states
            }
        else:
            intermediate_outputs = None

        return output, intermediate_outputs

class MinimalSamProjector(Projector):
    """
    A projector that performs minimal cross attention when the visual encoder is SAM.
    Args:
        input_dim (int): The number of input channels.
        output_dim (int): The number of output channels.
        projector_scale (float): The scale factor for the projector.
    Attributes:
        conv_feature (nn.Conv2d): Convolutional layer to process the input features.
    Methods:
        forward(hidden_states, conditions=None, output_intermediate=False):
            Forward pass of the projector.
            Args:
                hidden_states (torch.Tensor): Input tensor of shape (batch_size, 256, 64, 64).
                conditions (torch.Tensor, optional): Condition tensor of shape (batch_size, output_dim). Default is None.
                output_intermediate (bool, optional): Whether to output intermediate results. Default is False.
            Returns:
                Tuple[torch.Tensor, dict]: Output tensor and a dictionary of intermediate results (if requested).
    """
    # a projector performs minimal cross attention
    def __init__(self, input_dim, output_dim, projector_scale):
        super().__init__(input_dim, output_dim, projector_scale)
        self.conv_feature = nn.Conv2d(
            in_channels=input_dim, 
            out_channels=output_dim, 
            kernel_size=4, stride=4, padding=0)

    def forward(self, hidden_states, conditions = None, output_intermediate = False):
        # hidden_states: (batch_size, 256, 64, 64)
        # conditions: (batch_size, output_dim)

        bs, c, h, w = hidden_states.shape
        assert c == 256 and h == 64 and w == 64
        hidden_states = hidden_states.permute(0, 2, 3, 1) # (batch_size, 64, 64, 256)
        hidden_states = self.layernorm_feature(hidden_states)  # (batch_size, 64, 64, output_dim)
        hidden_states = hidden_states.permute(0, 3, 1, 2) # (batch_size, output_dim, 64, 64)
        hidden_states = self.conv_feature(hidden_states) # (batch_size, output_dim, 16, 16)
        hidden_states = hidden_states.permute(0, 2, 3, 1) # (batch_size, 16, 16, output_dim)
        hidden_states = hidden_states.view(bs, 256, self.output_dim).contiguous() # (batch_size, 256, output_dim)

        if conditions is not None:
            conditions = conditions.view(-1, self.projector_scale * conditions.shape[-1])
            conditions = self.layernorm_condition(conditions)
            conditions = self.linear_condition(conditions)  # (batch_size, output_dim)
            
            conditions_key = conditions # (batch_size, output_dim)
            attention = torch.einsum("bso,bo->bs", hidden_states, conditions_key).contiguous()

            conditions_value = conditions  # (batch_size, output_dim)

            value = torch.einsum("bs,bo-> bso", attention, conditions_value).contiguous()
            output =  value + hidden_states
        else:
            output = hidden_states
        
        output = output.view(-1, output.shape[-1])

        if output_intermediate and conditions is not None:
            intermediate_outputs = {
                "attention": attention,
                "value": value,
                "hidden_states": hidden_states
            }
        else:
            intermediate_outputs = None

        return output, intermediate_outputs

class MinimalCLIPProjector(Projector):
    """
    A projector that performs minimal cross attention when the visual encoder is the original CLIP encoder.
    Args:
        input_dim (int): The dimension of the input embeddings.
        output_dim (int): The dimension of the output embeddings.
        projector_scale (int): The scale factor for the projector.
        spatial_merge_size (int): The size of the spatial merge.
    Methods:
        thw_to_cutoff(grid_thw):
            Computes the cutoff positions and number of tokens for each image based on the grid dimensions.
        batchify_image_sequences(image_embeddings, grid_thw):
            Splits and pads image embeddings into batches based on the grid dimensions.
        serialize_image_batch(image_embeddings, num_tokens_for_each_image):
            Serializes the image embeddings back into a single tensor.
        forward(hidden_states, conditions=None, output_intermediate=False, grid_thw=None):
            Performs the forward pass of the projector, applying cross attention if conditions are provided.
    Attributes:
        spatial_merge_size (int): The size of the spatial merge.
    """
    # a projector performs minimal cross attention
    def __init__(self, input_dim, output_dim, projector_scale, spatial_merge_size):
        super().__init__(input_dim, output_dim, projector_scale)
        self.spatial_merge_size = spatial_merge_size

    def thw_to_cutoff(self, grid_thw):
        merge_size = self.spatial_merge_size
        merge_length = merge_size ** 2
        num_tokens_for_each_image = grid_thw.prod(dim = -1) // merge_length
        cutoff_positions = torch.cumsum(num_tokens_for_each_image, dim = 0)
        return cutoff_positions, num_tokens_for_each_image

    def batchify_image_sequences(self, image_embeddings, grid_thw):
        cutoff_positions, num_tokens_for_each_image = self.thw_to_cutoff(grid_thw)
        max_num_tokens = num_tokens_for_each_image.max()
        image_embeddings = torch.split(image_embeddings, num_tokens_for_each_image.tolist())
        image_embeddings = [F.pad(image_embedding, (0, 0, 0, max_num_tokens - image_embedding.shape[0]), value=0) for image_embedding in image_embeddings]
        return torch.stack(image_embeddings), num_tokens_for_each_image

    def serialize_image_batch(self, image_embeddings, num_tokens_for_each_image):
        # image_embeddings: (batch_size, max_num_tokens, embedding_dim)
        # num_tokens_for_each_image: (batch_size)
        batch_size = image_embeddings.shape[0]
        serialized_image_embeddings = []
        for i in range(batch_size):
            serialized_image_embeddings.append(image_embeddings[i][:num_tokens_for_each_image[i]])
        return torch.cat(serialized_image_embeddings, dim=0)

    def forward(self, hidden_states, conditions = None, output_intermediate = False, grid_thw = None):
        # hidden_states: (batch_size, seq_len, input_dim)
        # conditions: (batch_size, output_dim)

        if conditions is not None:
            assert grid_thw is not None
            conditions = conditions.view(-1, self.projector_scale * conditions.shape[-1])
            conditions = self.layernorm_condition(conditions)
            conditions = self.linear_condition(conditions)  # (batch_size, output_dim)

            hidden_states_batch, num_tokens_for_each_image = self.batchify_image_sequences(hidden_states, grid_thw)
            
            conditions_key = conditions # (batch_size, output_dim)
            attention = torch.einsum("bso,bo->bs", hidden_states_batch, conditions_key).contiguous()

            conditions_value = conditions  # (batch_size, output_dim)

            value = torch.einsum("bs,bo-> bso", attention, conditions_value).contiguous()
            value = self.serialize_image_batch(value, num_tokens_for_each_image)

            output =  value + hidden_states
        else:
            output = hidden_states
        
        output = output.view(-1, output.shape[-1])

        if output_intermediate and conditions is not None:
            intermediate_outputs = {
                "attention": attention,
                "value": value,
                "hidden_states": hidden_states
            }
        else:
            intermediate_outputs = None

        return output, intermediate_outputs

class VPT_Qwen2VLProcessor(Qwen2VLProcessor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detection_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.seg_processor = SamProcessor.from_pretrained("facebook/sam-vit-large")

class VPT_Qwen2VLConfig(Qwen2VLConfig):
    model_type = "qwen2_vl_vpt"

    def __init__(
            self, 
            detection_image_token_id=None, 
            detection_action_id=None,
            detection_action_start_id=None,
            para_start_id=None,
            para_end_id=None,
            clip_image_token_id=None, 
            clip_action_id=None,
            clip_action_start_id=None,
            seg_image_token_id=None, 
            seg_action_id=None,
            seg_action_start_id=None,
            num_inner_forward_run = 1,
            projector_scale = 1,
            para_mask_id = 0,
            para_mask_ratio = 1,
            alignment = False,
            vision_encoder_ls = [],
            **kwargs
        ):
        super().__init__(**kwargs)
        self.detection_image_token_id = detection_image_token_id
        self.detection_action_id = detection_action_id
        self.detection_action_start_id = detection_action_start_id
        self.para_start_id = para_start_id
        self.para_end_id = para_end_id
        self.clip_image_token_id = clip_image_token_id
        self.clip_action_id = clip_action_id
        self.clip_action_start_id = clip_action_start_id
        self.seg_image_token_id = seg_image_token_id
        self.seg_action_id = seg_action_id
        self.seg_action_start_id = seg_action_start_id
        self.num_inner_forward_run = num_inner_forward_run
        self.projector_scale = projector_scale
        self.para_mask_id = para_mask_id
        self.para_mask_ratio = para_mask_ratio
        self.alignment = alignment
        self.vision_encoder_ls = vision_encoder_ls

class VPT_Qwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    """
    VPT_Qwen2VLForConditionalGeneration is a model class for conditional generation tasks that extends the 
    Qwen2VLForConditionalGeneration class. This class integrates various vision encoders and projectors.
    
    Args:
        config (Config): Configuration object containing model parameters.
    Attributes:
        detection_model (Optional[PreTrainedModel]): The detection model used for processing detection images.
        detection_projector (Optional[MinimalProjector]): Projector for aligning detection model outputs with the main model.
        clip_projector (Optional[MinimalCLIPProjector]): Projector for aligning clip model outputs with the main model.
        seg_model (Optional[PreTrainedModel]): The segmentation model used for processing segmentation images.
        seg_projector (Optional[MinimalSamProjector]): Projector for aligning segmentation model outputs with the main model.
    Methods:
        thw_to_cutoff(grid_thw):
            Converts grid dimensions to cutoff positions for token sequences.
        batchify_image_sequences(image_embeddings, grid_thw):
            Batches image embeddings based on grid dimensions.
        serialize_image_batch(image_embeddings, num_tokens_for_each_image):
            Serializes batched image embeddings into a single tensor.
        mask_before_action_token(ids):
            Creates a mask for tokens before the action token.
        mask_para_detection(ids, para_id, ratio=1):
            Creates a mask for detection paragraphs based on the given ratio.
        mask_para_clip(ids, para_id, ratio=1):
            Creates a mask for clip paragraphs based on the given ratio.
        mask_para_seg(ids, para_id, ratio=1):
            Creates a mask for segmentation paragraphs based on the given ratio.
        forward(input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, 
                labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, 
                pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, rope_deltas=None, 
                detection_images=None, seg_images=None, clip_pixel_values=None, clip_images_grid_thw=None):
            Forward pass for the model. Handles various types of visual inputs and integrates them into the main model.
        correct_detection_action_token(input_ids):
            Corrects the detection action token in the input IDs.
        correct_clip_action_token(input_ids):
            Corrects the clip action token in the input IDs.
        correct_seg_action_token(input_ids):
            Corrects the segmentation action token in the input IDs.
        input_last_two_detection(input_ids, cache_position, past_key_values):
            Processes the last two detection tokens in the input IDs.
        input_last_two_clip(input_ids, cache_position, past_key_values):
            Processes the last two clip tokens in the input IDs.
        input_last_two_seg(input_ids, cache_position, past_key_values):
            Processes the last two segmentation tokens in the input IDs.
        prepare_inputs_for_generation(input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, 
                                      cache_position=None, position_ids=None, use_cache=True, pixel_values=None, 
                                      pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, 
                                      detection_images=None, clip_pixel_values=None, clip_images_grid_thw=None, 
                                      seg_images=None, **kwargs):
            Prepares inputs for the generation process, handling various types of visual inputs and cache positions.
    """
    config_class = VPT_Qwen2VLConfig

    def __init__(self, config):
        super().__init__(config)

        if "dino" in self.config.vision_encoder_ls:
            self.detection_model = AutoModel.from_config(AutoConfig.from_pretrained('facebook/dinov2-large'))
            self.detection_projector =  MinimalProjector(
                self.detection_model.config.hidden_size, self.config.hidden_size, config.projector_scale).to(device = self.detection_model.device)
        else:
            self.detection_model = None
            self.detection_projector = None
        
        if "clip" in self.config.vision_encoder_ls:
            self.clip_projector =  MinimalCLIPProjector(
                self.config.hidden_size, self.config.hidden_size, config.projector_scale, config.vision_config.spatial_merge_size).to(device = self.model.device)
        else:
            self.clip_projector = None

        if "sam" in self.config.vision_encoder_ls:
            self.seg_model = SamModel.from_pretrained("facebook/sam-vit-large").vision_encoder
            self.seg_projector =  MinimalSamProjector(
                self.seg_model.config.output_channels, self.config.hidden_size, config.projector_scale).to(device = self.model.device)
        else:
            self.seg_model = None
            self.seg_projector = None

    def thw_to_cutoff(self, grid_thw):
        merge_size = self.config.vision_config.spatial_merge_size
        merge_length = merge_size ** 2
        num_tokens_for_each_image = grid_thw.prod(dim = -1) // merge_length
        cutoff_positions = torch.cumsum(num_tokens_for_each_image, dim = 0)
        return cutoff_positions, num_tokens_for_each_image

    def batchify_image_sequences(self, image_embeddings, grid_thw):
        cutoff_positions, num_tokens_for_each_image = self.thw_to_cutoff(grid_thw)
        max_num_tokens = num_tokens_for_each_image.max()
        image_embeddings = torch.split(image_embeddings, num_tokens_for_each_image.tolist())
        image_embeddings = [F.pad(image_embedding, (0, 0, 0, max_num_tokens - image_embedding.shape[0]), value=0) for image_embedding in image_embeddings]
        return torch.stack(image_embeddings), num_tokens_for_each_image

    def serialize_image_batch(self, image_embeddings, num_tokens_for_each_image):
        # image_embeddings: (batch_size, max_num_tokens, embedding_dim)
        # num_tokens_for_each_image: (batch_size)
        batch_size = image_embeddings.shape[0]
        serialized_image_embeddings = []
        for i in range(batch_size):
            serialized_image_embeddings.append(image_embeddings[i][:num_tokens_for_each_image[i]])
        return torch.cat(serialized_image_embeddings, dim=0)

    def mask_before_action_token(self, ids):
        mask = ids == self.config.detection_action_start_id
        indices = torch.nonzero(mask, as_tuple=False)  # (row_idx, col_idx)
        indices_each_row = indices[:, 1:]
        before_mask = torch.arange(ids.shape[1], device = ids.device).unsqueeze(0)
        before_mask = before_mask.expand(ids.shape[0], ids.shape[1]) < indices_each_row
        return before_mask

    def mask_para_detection(self, ids, para_id, ratio = 1):
        action_token_mask = (ids == self.config.detection_action_id).any(dim=1)
        mask = torch.zeros_like(ids, dtype=torch.bool, device = ids.device)

        if action_token_mask.any() and torch.rand(1).item() < ratio:
            action_row_ids = ids[action_token_mask]

            para_start_positions = (action_row_ids == self.config.para_start_id).nonzero(as_tuple=False)
            para_end_positions = (action_row_ids == self.config.para_end_id).nonzero(as_tuple=False)

            start_count_zeros = (para_start_positions[:, 0] == 0).sum().item()
            end_count_zeros = (para_end_positions[:, 0] == 0).sum().item()

            para_start_positions = para_start_positions[:,1].view(-1,start_count_zeros)
            para_end_positions = para_end_positions[:,1].view(-1,end_count_zeros)

            mask_start = para_start_positions[:,para_id:para_id+1]
            mask_end = para_end_positions[:,para_id:para_id+1]
            action_row_mask = torch.arange(action_row_ids.shape[1], device = action_row_ids.device).unsqueeze(0).expand(action_row_ids.shape[0], action_row_ids.shape[1])
            action_row_mask = (action_row_mask > mask_start) & (action_row_mask < mask_end)

            mask[action_token_mask] = action_row_mask
        return mask

    def mask_para_clip(self, ids, para_id, ratio = 1):
        action_token_mask = (ids == self.config.clip_action_id).any(dim=1)
        mask = torch.zeros_like(ids, dtype=torch.bool, device = ids.device)

        if action_token_mask.any() and torch.rand(1).item() < ratio:
            action_row_ids = ids[action_token_mask]

            para_start_positions = (action_row_ids == self.config.para_start_id).nonzero(as_tuple=False)
            para_end_positions = (action_row_ids == self.config.para_end_id).nonzero(as_tuple=False)

            start_count_zeros = (para_start_positions[:, 0] == 0).sum().item()
            end_count_zeros = (para_end_positions[:, 0] == 0).sum().item()

            para_start_positions = para_start_positions[:,1].view(-1,start_count_zeros)
            para_end_positions = para_end_positions[:,1].view(-1,end_count_zeros)

            mask_start = para_start_positions[:,para_id:para_id+1]
            mask_end = para_end_positions[:,para_id:para_id+1]
            action_row_mask = torch.arange(action_row_ids.shape[1], device = action_row_ids.device).unsqueeze(0).expand(action_row_ids.shape[0], action_row_ids.shape[1])
            action_row_mask = (action_row_mask > mask_start) & (action_row_mask < mask_end)

            mask[action_token_mask] = action_row_mask
        return mask

    def mask_para_seg(self, ids, para_id, ratio = 1):
        action_token_mask = (ids == self.config.seg_action_id).any(dim=1)
        mask = torch.zeros_like(ids, dtype=torch.bool, device = ids.device)

        if action_token_mask.any() and torch.rand(1).item() < ratio:
            action_row_ids = ids[action_token_mask]

            para_start_positions = (action_row_ids == self.config.para_start_id).nonzero(as_tuple=False)
            para_end_positions = (action_row_ids == self.config.para_end_id).nonzero(as_tuple=False)

            start_count_zeros = (para_start_positions[:, 0] == 0).sum().item()
            end_count_zeros = (para_end_positions[:, 0] == 0).sum().item()

            para_start_positions = para_start_positions[:,1].view(-1,start_count_zeros)
            para_end_positions = para_end_positions[:,1].view(-1,end_count_zeros)

            mask_start = para_start_positions[:,para_id:para_id+1]
            mask_end = para_end_positions[:,para_id:para_id+1]
            action_row_mask = torch.arange(action_row_ids.shape[1], device = action_row_ids.device).unsqueeze(0).expand(action_row_ids.shape[0], action_row_ids.shape[1])
            action_row_mask = (action_row_mask > mask_start) & (action_row_mask < mask_end)

            mask[action_token_mask] = action_row_mask
        return mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        detection_images: Optional[torch.Tensor] = None,
        seg_images: Optional[torch.Tensor] = None,
        clip_pixel_values: Optional[torch.Tensor] = None,
        clip_images_grid_thw: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        """
        Perform a forward pass through the model.
        Args:
            input_ids (torch.LongTensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            position_ids (torch.LongTensor, optional): Position IDs.
            past_key_values (List[torch.FloatTensor], optional): Past key values for caching.
            inputs_embeds (torch.FloatTensor, optional): Input embeddings.
            labels (torch.LongTensor, optional): Labels for computing the loss.
            use_cache (bool, optional): Whether to use cache.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary.
            pixel_values (torch.Tensor, optional): Pixel values for image inputs.
            pixel_values_videos (torch.FloatTensor, optional): Pixel values for video inputs.
            image_grid_thw (torch.LongTensor, optional): Image grid dimensions.
            video_grid_thw (torch.LongTensor, optional): Video grid dimensions.
            rope_deltas (torch.LongTensor, optional): Rope deltas.
            detection_images (torch.Tensor, optional): Images for detection model.
            seg_images (torch.Tensor, optional): Images for Segmentation model.
            clip_pixel_values (torch.Tensor, optional): Additional images for the original CLIP model.
            clip_images_grid_thw (torch.LongTensor, optional): Additional images for the original CLIP model.
        Returns:
            Union[Tuple, Qwen2VLCausalLMOutputWithPast]: Model outputs.
        """

        run_times = self.config.num_inner_forward_run

        if run_times == 2:
            pass
        elif run_times == 1:
            pass
        else:
            raise ValueError("Number of inner forward runs should be 1 or 2.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        with torch.no_grad():
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)

            if pixel_values is not None:
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if detection_images is not None:
                detection_image_mask = (input_ids == self.config.detection_image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                detection_embedding = self.detection_model(pixel_values = detection_images)[0][:,1:]

            if seg_images is not None:
                seg_image_mask = (input_ids == self.config.seg_image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                seg_images = seg_images.to(dtype = inputs_embeds.dtype)
                seg_embedding = self.seg_model(pixel_values = seg_images)[0]

            if clip_pixel_values is not None:
                clip_image_mask = (input_ids == self.config.clip_image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                clip_pixel_values = clip_pixel_values.type(self.visual.get_dtype())
                clip_embedding = self.visual(clip_pixel_values, grid_thw=clip_images_grid_thw)
                clip_embedding = clip_embedding.to(inputs_embeds.device, inputs_embeds.dtype)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if run_times == 2:
            inputs_embeds_first_round = inputs_embeds.clone()
            if detection_images is not None and clip_pixel_values is None and seg_images is None:
                detection_embedding_first_round = detection_embedding.clone()
                detection_embedding_first_round, _ = self.detection_projector(detection_embedding_first_round)
                detection_embedding_first_round = detection_embedding_first_round.view(-1, detection_embedding_first_round.shape[-1])
                detection_embedding_first_round = detection_embedding_first_round.to(detection_embedding_first_round.device, detection_embedding_first_round.dtype)
                detection_embedding_first_round = detection_embedding_first_round.contiguous()
                inputs_embeds_first_round = inputs_embeds_first_round.masked_scatter(detection_image_mask, detection_embedding_first_round)

            elif detection_images is None and clip_pixel_values is not None and seg_images is None:
                clip_embedding_first_round = clip_embedding.clone()
                clip_embedding_first_round, _ = self.clip_projector(clip_embedding_first_round)
                clip_embedding_first_round = clip_embedding_first_round.view(-1, clip_embedding_first_round.shape[-1])
                clip_embedding_first_round = clip_embedding_first_round.to(inputs_embeds_first_round.device, inputs_embeds_first_round.dtype)
                clip_embedding_first_round = clip_embedding_first_round.contiguous()
                inputs_embeds_first_round = inputs_embeds_first_round.masked_scatter(clip_image_mask, clip_embedding_first_round)

            elif detection_images is None and clip_pixel_values is None and seg_images is not None:
                seg_embedding_first_round = seg_embedding.clone()
                seg_embedding_first_round, _ = self.seg_projector(seg_embedding_first_round)
                seg_embedding_first_round = seg_embedding_first_round.view(-1, seg_embedding_first_round.shape[-1])
                seg_embedding_first_round = seg_embedding_first_round.to(inputs_embeds_first_round.device, inputs_embeds_first_round.dtype)
                seg_embedding_first_round = seg_embedding_first_round.contiguous()
                inputs_embeds_first_round = inputs_embeds_first_round.masked_scatter(seg_image_mask, seg_embedding_first_round)

            outputs_first_round = self.model(
                input_ids=None,
                position_ids=position_ids.clone() if position_ids is not None else None,
                attention_mask=attention_mask.clone() if attention_mask is not None else None,
                past_key_values=copy.deepcopy(past_key_values) if past_key_values is not None else None,
                inputs_embeds=inputs_embeds_first_round,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_state_first_round = outputs_first_round[0]
            shift_hidden_state_first_round = hidden_state_first_round[..., :-1, :].contiguous().clone()
            shift_input_ids = input_ids[..., 1:].contiguous().clone()

            if detection_images is not None and clip_pixel_values is None and seg_images is None:
                if (shift_input_ids == self.config.detection_action_id).sum() == 0:
                    assert False, "Detection action token not found in the input."
                else:
                    detection_token_mask = (shift_input_ids == self.config.detection_action_id).unsqueeze(-1).expand_as(shift_hidden_state_first_round)
                    detection_prompt = shift_hidden_state_first_round.masked_select(detection_token_mask).view(-1, shift_hidden_state_first_round.shape[-1])

                detection_embedding, _ = self.detection_projector(detection_embedding, detection_prompt, output_intermediate = False)

                detection_embedding = detection_embedding.view(-1, detection_embedding.shape[-1])
                detection_embedding = detection_embedding.to(detection_embedding.device, detection_embedding.dtype)
                detection_embedding = detection_embedding.contiguous()
                inputs_embeds = inputs_embeds.masked_scatter(detection_image_mask, detection_embedding)

                if self.config.para_mask_id == 0:
                    pass
                elif self.config.para_mask_id == 1:
                    para_mask = self.mask_para_detection(input_ids, 1, self.config.para_mask_ratio)
                    para_mask = para_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_fill(para_mask, 0)
                elif self.config.para_mask_id == 2:
                    para_mask = self.mask_para_detection(input_ids, 1, self.config.para_mask_ratio) | self.mask_para_detection(input_ids, 3, self.config.para_mask_ratio)
                    para_mask = para_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_fill(para_mask, 0)
                else:
                    raise ValueError("Invalid para_mask value.")

                action_token_mask = (input_ids == self.config.detection_action_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(action_token_mask, detection_prompt)

            elif detection_images is None and clip_pixel_values is not None and seg_images is None:
                if (shift_input_ids == self.config.clip_action_id).sum() == 0:
                    assert False, "clip action token not found in the input."
                else:
                    clip_token_mask = (shift_input_ids == self.config.clip_action_id).unsqueeze(-1).expand_as(shift_hidden_state_first_round)
                    clip_prompt = shift_hidden_state_first_round.masked_select(clip_token_mask).view(-1, shift_hidden_state_first_round.shape[-1])

                clip_embedding, _ = self.clip_projector(clip_embedding, clip_prompt, output_intermediate = False, grid_thw = clip_images_grid_thw)

                clip_embedding = clip_embedding.view(-1, clip_embedding.shape[-1])
                clip_embedding = clip_embedding.to(inputs_embeds.device, inputs_embeds.dtype)
                clip_embedding = clip_embedding.contiguous()
                inputs_embeds = inputs_embeds.masked_scatter(clip_image_mask, clip_embedding)

                if self.config.para_mask_id == 0:
                    pass
                elif self.config.para_mask_id == 1:
                    para_mask = self.mask_para_clip(input_ids, 1, self.config.para_mask_ratio)
                    para_mask = para_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_fill(para_mask, 0)
                elif self.config.para_mask_id == 2:
                    para_mask = self.mask_para_clip(input_ids, 1, self.config.para_mask_ratio) | self.mask_para_clip(input_ids, 3, self.config.para_mask_ratio)
                    para_mask = para_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_fill(para_mask, 0)
                else:
                    raise ValueError("Invalid para_mask value.")

                action_token_mask = (input_ids == self.config.clip_action_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(action_token_mask, clip_prompt)

            elif detection_images is None and clip_pixel_values is None and seg_images is not None:
                if (shift_input_ids == self.config.seg_action_id).sum() == 0:
                    assert False, "seg action token not found in the input."
                else:
                    seg_token_mask = (shift_input_ids == self.config.seg_action_id).unsqueeze(-1).expand_as(shift_hidden_state_first_round)
                    seg_prompt = shift_hidden_state_first_round.masked_select(seg_token_mask).view(-1, shift_hidden_state_first_round.shape[-1])

                seg_embedding, _ = self.seg_projector(seg_embedding, seg_prompt, output_intermediate = False)
                seg_embedding = seg_embedding.view(-1, seg_embedding.shape[-1])
                seg_embedding = seg_embedding.to(inputs_embeds.device, inputs_embeds.dtype)
                seg_embedding = seg_embedding.contiguous()
                inputs_embeds = inputs_embeds.masked_scatter(seg_image_mask, seg_embedding)

                if self.config.para_mask_id == 0:
                    pass
                elif self.config.para_mask_id == 1:
                    para_mask = self.mask_para_seg(input_ids, 1, self.config.para_mask_ratio)
                    para_mask = para_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_fill(para_mask, 0)
                elif self.config.para_mask_id == 2:
                    para_mask = self.mask_para_seg(input_ids, 1, self.config.para_mask_ratio) | self.mask_para_seg(input_ids, 3, self.config.para_mask_ratio)
                    para_mask = para_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_fill(para_mask, 0)
                else:
                    raise ValueError("Invalid para_mask value.")

                action_token_mask = (input_ids == self.config.seg_action_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(action_token_mask, seg_prompt)

            elif detection_images is None and clip_pixel_values is None and seg_images is None:
                if self.config.detection_action_id is None:
                    pass
                else:
                    if (shift_input_ids == self.config.detection_action_id).sum() == 0:
                        pass
                    else:
                        detection_token_mask = (shift_input_ids == self.config.detection_action_id).unsqueeze(-1).expand_as(shift_hidden_state_first_round)
                        detection_prompt = shift_hidden_state_first_round.masked_select(detection_token_mask).view(-1, shift_hidden_state_first_round.shape[-1])
                        action_token_mask = (input_ids == self.config.detection_action_id).unsqueeze(-1).expand_as(inputs_embeds)
                        inputs_embeds = inputs_embeds.masked_scatter(action_token_mask, detection_prompt)
                if self.config.clip_action_id is None:
                    pass
                else:
                    if (shift_input_ids == self.config.clip_action_id).sum() == 0:
                        pass
                    else:
                        clip_token_mask = (shift_input_ids == self.config.clip_action_id).unsqueeze(-1).expand_as(shift_hidden_state_first_round)
                        clip_prompt = shift_hidden_state_first_round.masked_select(clip_token_mask).view(-1, shift_hidden_state_first_round.shape[-1])
                        action_token_mask = (input_ids == self.config.clip_action_id).unsqueeze(-1).expand_as(inputs_embeds)
                        inputs_embeds = inputs_embeds.masked_scatter(action_token_mask, clip_prompt)
                if self.config.seg_action_id is None:
                    pass
                else:
                    if (shift_input_ids == self.config.seg_action_id).sum() == 0:
                        pass
                    else:
                        seg_token_mask = (shift_input_ids == self.config.seg_action_id).unsqueeze(-1).expand_as(shift_hidden_state_first_round)
                        seg_prompt = shift_hidden_state_first_round.masked_select(seg_token_mask).view(-1, shift_hidden_state_first_round.shape[-1])
                        action_token_mask = (input_ids == self.config.seg_action_id).unsqueeze(-1).expand_as(inputs_embeds)
                        inputs_embeds = inputs_embeds.masked_scatter(action_token_mask, seg_prompt)
            else:
                assert False, "Invalid input."

            outputs = self.model(
                    input_ids=None,
                    position_ids=position_ids.clone() if position_ids is not None else None,
                    attention_mask=attention_mask.clone() if attention_mask is not None else None,
                    past_key_values=copy.deepcopy(past_key_values) if past_key_values is not None else None,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        elif run_times == 1:
            if detection_images is not None and clip_pixel_values is None and seg_images is None:
                detection_embedding, _ = self.detection_projector(detection_embedding)
                detection_embedding = detection_embedding.view(-1, detection_embedding.shape[-1])
                detection_embedding = detection_embedding.to(inputs_embeds.device, inputs_embeds.dtype)
                detection_embedding = detection_embedding.contiguous()
                inputs_embeds = inputs_embeds.masked_scatter(detection_image_mask, detection_embedding)
            
            elif detection_images is None and clip_pixel_values is not None and seg_images is None:
                clip_embedding, _ = self.clip_projector(clip_embedding)
                clip_embedding = clip_embedding.view(-1, clip_embedding.shape[-1])
                clip_embedding = clip_embedding.to(inputs_embeds.device, inputs_embeds.dtype)
                clip_embedding = clip_embedding.contiguous()
                inputs_embeds = inputs_embeds.masked_scatter(clip_image_mask, clip_embedding)

            elif detection_images is None and clip_pixel_values is None and seg_images is not None:
                seg_embedding, _ = self.seg_projector(seg_embedding)
                seg_embedding = seg_embedding.view(-1, seg_embedding.shape[-1])
                seg_embedding = seg_embedding.to(inputs_embeds.device, inputs_embeds.dtype)
                seg_embedding = seg_embedding.contiguous()
                inputs_embeds = inputs_embeds.masked_scatter(seg_image_mask, seg_embedding)

            outputs = self.model(
                input_ids=None,
                position_ids=position_ids.clone() if position_ids is not None else None,
                attention_mask=attention_mask.clone() if attention_mask is not None else None,
                past_key_values=copy.deepcopy(past_key_values) if past_key_values is not None else None,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        else:
            raise ValueError("Number of inner forward runs should be 1 or 2.")

        # self.check_validity(outputs[0])

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    def correct_detection_action_token(self, input_ids):
        if self.config.detection_action_start_id is None:
            return input_ids
        else:
            mask = input_ids == self.config.detection_action_start_id
            if mask.any():
                indices = torch.nonzero(mask, as_tuple=False)  # (row_idx, col_idx)
                next_indices = indices.clone()
                next_indices[:, 1] += 1 
                next_indices = next_indices[next_indices[:, 1] < input_ids.size(1)]
                input_ids[next_indices[:, 0], next_indices[:, 1]] = self.config.detection_action_id
            return input_ids

    def correct_clip_action_token(self, input_ids):
        if self.config.clip_action_start_id is None:
            return input_ids
        else:
            mask = input_ids == self.config.clip_action_start_id
            if mask.any():
                indices = torch.nonzero(mask, as_tuple=False)  # (row_idx, col_idx)
                next_indices = indices.clone()
                next_indices[:, 1] += 1 
                next_indices = next_indices[next_indices[:, 1] < input_ids.size(1)]
                input_ids[next_indices[:, 0], next_indices[:, 1]] = self.config.clip_action_id
            return input_ids

    def correct_seg_action_token(self, input_ids):
        if self.config.seg_action_start_id is None:
            return input_ids
        else:
            mask = input_ids == self.config.seg_action_start_id
            if mask.any():
                indices = torch.nonzero(mask, as_tuple=False)  # (row_idx, col_idx)
                next_indices = indices.clone()
                next_indices[:, 1] += 1 
                next_indices = next_indices[next_indices[:, 1] < input_ids.size(1)]
                input_ids[next_indices[:, 0], next_indices[:, 1]] = self.config.seg_action_id
            return input_ids

    def input_last_two_detection(self, input_ids, cache_position, past_key_values):
        if self.config.detection_action_id is None:
            input_ids = input_ids[:, cache_position]
            return input_ids, cache_position, past_key_values
        else:
            mask = input_ids[:,-1] == self.config.detection_action_id
            if mask.any():
                input_ids = input_ids[:, -2:]
                cache_position = torch.cat([cache_position-1, cache_position])
                past_key_values.crop(-1)
            else:
                input_ids = input_ids[:, cache_position]
            return input_ids, cache_position, past_key_values

    def input_last_two_clip(self, input_ids, cache_position, past_key_values):
        if self.config.clip_action_id is None:
            input_ids = input_ids[:, cache_position]
            return input_ids, cache_position, past_key_values
        else:
            mask = input_ids[:,-1] == self.config.clip_action_id
            if mask.any():
                input_ids = input_ids[:, -2:]
                cache_position = torch.cat([cache_position-1, cache_position])
                past_key_values.crop(-1)
            else:
                input_ids = input_ids[:, cache_position]
            return input_ids, cache_position, past_key_values

    def input_last_two_seg(self, input_ids, cache_position, past_key_values):
        if self.config.seg_action_id is None:
            input_ids = input_ids[:, cache_position]
            return input_ids, cache_position, past_key_values
        else:
            mask = input_ids[:,-1] == self.config.seg_action_id
            if mask.any():
                input_ids = input_ids[:, -2:]
                cache_position = torch.cat([cache_position-1, cache_position])
                past_key_values.crop(-1)
            else:
                input_ids = input_ids[:, cache_position]
            return input_ids, cache_position, past_key_values

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        detection_images = None,
        clip_pixel_values = None,
        clip_images_grid_thw = None,
        seg_images = None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                ## Attention ## manually control if needed
                if self.config.detection_action_id is not None and self.config.clip_action_id is None and self.config.seg_action_id is None:
                    input_ids = self.correct_detection_action_token(input_ids)
                    input_ids, cache_position, past_key_values = self.input_last_two_detection(input_ids, cache_position, past_key_values)
                elif self.config.detection_action_id is None and self.config.clip_action_id is not None and self.config.seg_action_id is None:
                    input_ids = self.correct_clip_action_token(input_ids)
                    input_ids, cache_position, past_key_values = self.input_last_two_clip(input_ids, cache_position, past_key_values)
                elif self.config.detection_action_id is None and self.config.clip_action_id is None and self.config.seg_action_id is not None:
                    input_ids = self.correct_seg_action_token(input_ids)
                    input_ids, cache_position, past_key_values = self.input_last_two_seg(input_ids, cache_position, past_key_values)
                else:
                    input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None
            detection_images = None
            seg_images = None
            clip_pixel_values = None
            clip_images_grid_thw = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
                "detection_images": detection_images,
                "seg_images": seg_images,
                "clip_pixel_values": clip_pixel_values,
                "clip_images_grid_thw": clip_images_grid_thw,
            }
        )
        return model_inputs