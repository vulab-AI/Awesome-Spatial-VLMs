# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .modeling_internlm2 import InternLM2ForCausalLM

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

# def extract_local(value, rank, world_size, dim=1):
#     """Extract local tensor across the sequence dimension."""
#     value_chunks = value.chunk(2 * world_size, dim=dim)
#     local_value = torch.cat(
#         [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
#     )
#     return local_value.to(value.device)
def extract_local(value, cu_seqlens, rank, world_size, dim=1):
    local_values = []
    value = value.transpose(0, dim)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        local_value = value[start:end].chunk(world_size, dim=0)[rank].detach().clone()
        local_values.append(local_value)
    return torch.cat(local_values, dim=0).transpose(0, dim).contiguous()

def extract_local2(value, rank, world_size,  dim=1):
    """Extract local tensor across the hidden dimension."""
    dimension_size = value.shape[dim]
    sub_seq_length = dimension_size // world_size

    sub_seq_start = rank * sub_seq_length
    sub_seq_end = (rank + 1) * sub_seq_length
    local_value = value[:, sub_seq_start:sub_seq_end]

    return local_value.to(value.device)
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size(local_group))]
        dist.all_gather(output, input, group=local_group)
        return torch.stack(output, 0)

    @staticmethod
    def backward(ctx, grads):
        (input,) = ctx.saved_tensors
        dist.all_reduce(grads, group=local_group)
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank(local_group)]
        return grad_out
    

class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    # main_input_name = 'pixel_values'
    main_input_name = 'input_ids'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True, use_ring_flash_attn=False, group_list=None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.enable_cross_attention = config.enable_cross_attention
        self.enable_shared_cross_attention = config.enable_shared_cross_attention
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        if config._attn_implementation is not None and config._attn_implementation not in ['eager', 'flash_attention_2', 'sdpa']:
            config.llm_config.attn_implementation = config._attn_implementation
        if config._attn_implementation == 'eager':
            config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
        print(f"LLM Using {config.llm_config.attn_implementation} attention implementation")
        self.use_flash_attn = use_flash_attn
        self.use_ring_flash_attn = use_ring_flash_attn
        if use_ring_flash_attn:
            config.llm_config.attn_implementation = 'ring_flash_attn'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')
        self.group_list = group_list
        self.language_model.group_list = group_list
        self.language_model.model.group_list = group_list

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.img_start_token_id = None
        self.img_end_token_id = None
        self.bos_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        global local_group
        if self.group_list is not None:
            for group_idx,group in enumerate(self.group_list):
                if type(group)==torch.distributed.distributed_c10d.ProcessGroup:
                    # assert type(group)==torch.distributed.distributed_c10d.ProcessGroup
                    break        # print("Printing decoded input ids")
            local_group=group
        else:
            local_group=None
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # print(f"pixel_values.shape: {pixel_values.shape if pixel_values is not None else None}")
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        encoder_hidden_states = None
        packed_encoder_position_ids = encoder_position_ids # only use when packing data for training
        packed_encoder_attention_mask = encoder_attention_mask # only use when packing data for training
        encoder_attention_mask = None
        encoder_position_ids = None
        if pixel_values is not None:
            if image_flags is not None:
                image_flags = image_flags.squeeze(-1)
            else:
                image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long, device=pixel_values.device)
                
            # extract using vit embeds
            vit_embeds = self.extract_feature(pixel_values)
            if self.use_ring_flash_attn:
                group_size = dist.get_world_size(local_group)
                rank = dist.get_rank(local_group)
                img_num_dim = 0
                pad_num=0
                if pixel_values.shape[img_num_dim] > group_size:
                    if pixel_values.shape[img_num_dim] % group_size!=0:
                        pad_num = group_size - pixel_values.shape[img_num_dim] % group_size
                        if pad_num < group_size:  
                            pad_shape = list(pixel_values.shape)
                            pad_shape[img_num_dim] = pad_num  
                            pad_pixel = torch.zeros(pad_shape, dtype=pixel_values.dtype, device=pixel_values.device)

                            pixel_values = torch.cat([pixel_values, pad_pixel], dim=img_num_dim)

                    chunked_pixel=torch.chunk(pixel_values, group_size, dim=img_num_dim)
                    local_pixel=chunked_pixel[dist.get_rank(local_group)]
                    local_vit_embeds=self.extract_feature(local_pixel)
                    vit_embeds=GatherLayer.apply(local_vit_embeds)
                    vit_embeds=vit_embeds.view(-1,vit_embeds.shape[-2],vit_embeds.shape[-1])
                    if pad_num>0:
                        vit_embeds=vit_embeds[:-pad_num]
                else:
                    vit_embeds = self.extract_feature(pixel_values)
            else:
                vit_embeds = self.extract_feature(pixel_values)
            
            
            vit_embeds = vit_embeds[image_flags == 1]
            vit_batch_size = pixel_values.shape[0]

            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}, vit tokens: {vit_batch_size * 256}')

            if not self.enable_cross_attention:
                input_ids = input_ids.reshape(B * N)
                selected = (input_ids == self.img_context_token_id)
                try:
                    input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
                except Exception as e:
                    vit_embeds = vit_embeds.reshape(-1, C)
                    print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                        f'vit_embeds.shape={vit_embeds.shape}')
                    n_token = selected.sum()
                    input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            else:
                num_images, num_tokens_per_image, C = vit_embeds.shape
                num_imgs_per_sample = input_ids.eq(self.img_context_token_id).sum(dim=1)
                assert num_imgs_per_sample.sum() != 0
                max_num_img = max(num_imgs_per_sample)
                vit_embeds_per_sample = []
                encoder_attention_mask = torch.zeros((B, max_num_img, num_tokens_per_image), device=input_ids.device, dtype=input_ids.dtype)
                idx = 0
                for bz_i, num_img in enumerate(num_imgs_per_sample):
                    if max_num_img == num_img:
                        vit_embeds_per_sample.append(vit_embeds[idx:idx + num_img])
                    else:
                        vit_embeds_per_sample.append(
                            torch.cat(
                                [vit_embeds[idx:idx + num_img],
                                    torch.zeros_like(vit_embeds[0]).unsqueeze(0).expand(max_num_img - num_img, -1, -1)],
                                dim=0
                            )
                        )
                    encoder_attention_mask[bz_i, :num_img] = 1
                    idx += num_img
                vit_embeds = torch.stack(vit_embeds_per_sample, dim=0)
                encoder_hidden_states = vit_embeds.reshape(B, -1, C)
                encoder_attention_mask = encoder_attention_mask.reshape(B, -1)
                encoder_position_ids = torch.arange(
                    encoder_hidden_states.shape[1], dtype=torch.long, device=encoder_hidden_states.device
                ).unsqueeze(0).expand(B, -1)
                if packed_encoder_position_ids is not None:
                    assert packed_encoder_position_ids.shape == encoder_position_ids.shape, f'{packed_encoder_position_ids.shape} != {encoder_position_ids.shape}'
                    encoder_position_ids = packed_encoder_position_ids
                if packed_encoder_attention_mask is not None:
                    if not packed_encoder_attention_mask.shape[0] == encoder_attention_mask.shape[0] and \
                        packed_encoder_attention_mask.shape[-1] == encoder_attention_mask.shape[-1]:
                        raise ValueError(f'{packed_encoder_attention_mask.shape} != {encoder_attention_mask.shape}')
                    encoder_attention_mask = packed_encoder_attention_mask
            
            if self.enable_shared_cross_attention:
                # select the vit part as the encoder hidden states
                selected = (input_ids == self.img_context_token_id) | (input_ids == self.img_start_token_id) | (input_ids == self.img_end_token_id)
                all_encoder_hidden_states = []
                all_text_input_embeds = []
                all_text_attention_mask = []
                all_encoder_attention_mask = []
                for idx in range(B):
                    b_selected = selected[idx]
                    encoder_hidden_states = input_embeds[idx][b_selected]
                    text_input_embeds = input_embeds[idx][~b_selected]
                    if attention_mask.dim() == 2:
                        encoder_attention_mask = attention_mask[idx][b_selected]
                        text_attention_mask = attention_mask[idx][~b_selected]
                    elif attention_mask.dim() == 4:
                        encoder_attention_mask = attention_mask[idx][:, ~b_selected][..., b_selected].reshape(1, len(text_input_embeds), len(encoder_hidden_states))
                        text_attention_mask = attention_mask[idx][:, :, ~b_selected][..., ~b_selected].reshape(1, len(text_input_embeds), len(text_input_embeds))
                    else:
                        raise NotImplementedError(f'attention_mask.dim()={attention_mask.dim()}')
                    all_encoder_hidden_states.append(encoder_hidden_states)
                    all_text_input_embeds.append(text_input_embeds)
                    all_text_attention_mask.append(text_attention_mask)
                    all_encoder_attention_mask.append(encoder_attention_mask)
                
                # padding
                max_q_len = max([len(text_input_embeds) for text_input_embeds in all_text_input_embeds])
                max_k_len = max([len(encoder_hidden_states) for encoder_hidden_states in all_encoder_hidden_states])
                for idx in range(B):
                    all_encoder_hidden_states[idx] = torch.cat([
                        all_encoder_hidden_states[idx],
                        torch.zeros(max_k_len - len(all_encoder_hidden_states[idx]), C, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                    ], dim=0) if len(all_encoder_hidden_states[idx]) < max_k_len else all_encoder_hidden_states[idx]
                    all_text_input_embeds[idx] = torch.cat([
                        all_text_input_embeds[idx],
                        torch.zeros(max_q_len - len(all_text_input_embeds[idx]), C, device=text_input_embeds.device, dtype=text_input_embeds.dtype)
                    ], dim=0) if len(all_text_input_embeds[idx]) < max_q_len else all_text_input_embeds[idx]
                    if attention_mask.dim() == 2:
                        all_encoder_attention_mask[idx] = torch.cat([
                            all_encoder_attention_mask[idx],
                            torch.zeros(max_k_len - len(all_encoder_attention_mask[idx]), device=encoder_attention_mask.device, dtype=encoder_attention_mask.dtype)
                        ], dim=0) if len(all_encoder_attention_mask[idx]) < max_k_len else all_encoder_attention_mask[idx]
                        all_text_attention_mask[idx] = torch.cat([
                            all_text_attention_mask[idx],
                            torch.zeros(max_q_len - len(all_text_attention_mask[idx]), device=text_attention_mask.device, dtype=text_attention_mask.dtype)
                        ], dim=0) if len(all_text_attention_mask[idx]) < max_q_len else all_text_attention_mask[idx]
                    elif attention_mask.dim() == 4:
                        padding_size = all_encoder_attention_mask[idx].shape
                        padding_size = (padding_size[0], max_q_len - padding_size[1], padding_size[2])
                        all_encoder_attention_mask[idx] = torch.cat([
                                all_encoder_attention_mask[idx],
                                torch.zeros(padding_size, device=encoder_attention_mask.device, dtype=encoder_attention_mask.dtype)
                            ], dim=1) if all_encoder_attention_mask[idx].shape[1] < max_k_len else all_encoder_attention_mask[idx]
                        padding_size = all_encoder_attention_mask[idx].shape
                        padding_size = (padding_size[0], padding_size[1], max_k_len - padding_size[2])
                        all_encoder_attention_mask[idx] = torch.cat([
                                all_encoder_attention_mask[idx],
                                torch.zeros(padding_size, device=encoder_attention_mask.device, dtype=encoder_attention_mask.dtype)
                            ], dim=2) if all_encoder_attention_mask[idx].shape[2] < max_k_len else all_encoder_attention_mask[idx]
                        
                        padding_size = all_text_attention_mask[idx].shape
                        padding_size = (padding_size[0], max_q_len - padding_size[1], padding_size[2])
                        all_text_attention_mask[idx] = torch.cat([
                                all_text_attention_mask[idx],
                                torch.zeros(padding_size, device=text_attention_mask.device, dtype=text_attention_mask.dtype)
                            ], dim=1) if all_text_attention_mask[idx].shape[1] < max_q_len else all_text_attention_mask[idx]
                        padding_size = all_text_attention_mask[idx].shape
                        padding_size = (padding_size[0], padding_size[1], max_q_len - padding_size[2])
                        all_text_attention_mask[idx] = torch.cat([
                                all_text_attention_mask[idx],
                                torch.zeros(padding_size, device=text_attention_mask.device, dtype=text_attention_mask.dtype)
                            ], dim=2) if all_text_attention_mask[idx].shape[2] < max_q_len else all_text_attention_mask[idx]
                    else:
                        raise NotImplementedError(f'attention_mask.dim()={attention_mask.dim()}')
                encoder_hidden_states = torch.stack(all_encoder_hidden_states, dim=0)
                input_embeds = torch.stack(all_text_input_embeds, dim=0)
                encoder_attention_mask = torch.stack(all_encoder_attention_mask, dim=0)
                attention_mask = torch.stack(all_text_attention_mask, dim=0)
                
            input_embeds = input_embeds.reshape(B, N, C)
            
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            position_ids=position_ids,
            encoder_position_ids=encoder_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )
        logits = outputs.logits
        
        loss = outputs.loss
        # loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
        #     shift_labels = shift_labels.view(-1)
        #     # Enable model parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)

        # print(f"original loss: {loss}")
        # rank = dist.get_rank(local_group)
        # print(f"rank: {rank} reaching barrier")
        # dist.barrier()
        # print(f"rank: {rank} passed barrier")
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            if not self.enable_cross_attention:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            else:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            benchmark_efficiency: Optional[bool] = False,
            **generate_kwargs,
    ) -> torch.LongTensor:
        
        if benchmark_efficiency:
            import time
            efficiency_metrics = {
                "total_vit_forward_time": 0.0,
                "total_prefill_time": 0.0,
                "total_decoding_time": 0.0,
                "vit_forward_time_per_image": 0.0,
                "prefill_time_per_token": 0.0,
                "decoding_time_per_token": 0.0,
            }

            start = time.time()
        assert self.img_context_token_id is not None
        encoder_hidden_states = None
        encoder_attention_mask = None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape

            if not self.enable_cross_attention:
                input_embeds = input_embeds.reshape(B * N, C)
                input_ids = input_ids.reshape(B * N)
                selected = (input_ids == self.img_context_token_id)
                assert selected.sum() != 0
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

                input_embeds = input_embeds.reshape(B, N, C)
                input_ids = input_ids.reshape(B, N)
            else:
                num_images, num_tokens_per_image, C = vit_embeds.shape
                num_imgs_per_sample = input_ids.eq(self.img_context_token_id).sum(dim=1)
                assert num_imgs_per_sample.sum() != 0
                max_num_img = max(num_imgs_per_sample)
                vit_embeds_per_sample = []
                encoder_attention_mask = torch.zeros((B, max_num_img, num_tokens_per_image), device=input_ids.device, dtype=input_ids.dtype)
                idx = 0
                for num_img in num_imgs_per_sample:
                    if max_num_img == num_img:
                        vit_embeds_per_sample.append(vit_embeds[idx:idx + num_img])
                    else:
                        vit_embeds_per_sample.append(
                            torch.cat(
                                [vit_embeds[idx:idx + num_img],
                                    torch.zeros_like(vit_embeds[0]).unsqueeze(0).expand(max_num_img - num_img, -1, -1)],
                                dim=0
                            )
                        )
                    encoder_attention_mask[idx, :num_img] = 1
                    idx += num_img
                vit_embeds = torch.stack(vit_embeds_per_sample, dim=0)
                encoder_hidden_states = vit_embeds.reshape(B, -1, C)
                encoder_attention_mask = encoder_attention_mask.reshape(B, -1)
            
            if self.enable_shared_cross_attention:
                # select the vit part as the encoder hidden states
                selected = (input_ids == self.img_context_token_id) | (input_ids == self.img_start_token_id) | (input_ids == self.img_end_token_id) | (input_ids == self.bos_token_id)
                all_encoder_hidden_states = []
                all_text_input_embeds = []
                all_text_attention_mask = []
                all_encoder_attention_mask = []
                for idx in range(B):
                    b_selected = selected[idx]
                    encoder_hidden_states = input_embeds[idx][b_selected]
                    text_input_embeds = input_embeds[idx][~b_selected]
                    if attention_mask.dim() == 2:
                        encoder_attention_mask = attention_mask[idx][b_selected]
                        text_attention_mask = attention_mask[idx][~b_selected]
                    elif attention_mask.dim() == 4:
                        encoder_attention_mask = attention_mask[idx][:, ~b_selected][..., b_selected].reshape(1, len(text_input_embeds), len(encoder_hidden_states))
                        text_attention_mask = attention_mask[idx][:, :, ~b_selected][..., ~b_selected].reshape(1, len(text_input_embeds), len(text_input_embeds))
                    else:
                        raise NotImplementedError(f'attention_mask.dim()={attention_mask.dim()}')
                    all_encoder_hidden_states.append(encoder_hidden_states)
                    all_text_input_embeds.append(text_input_embeds)
                    all_text_attention_mask.append(text_attention_mask)
                    all_encoder_attention_mask.append(encoder_attention_mask)
                
                # padding
                max_q_len = max([len(text_input_embeds) for text_input_embeds in all_text_input_embeds])
                max_k_len = max([len(encoder_hidden_states) for encoder_hidden_states in all_encoder_hidden_states])
                for idx in range(B):
                    all_encoder_hidden_states[idx] = torch.cat([
                        all_encoder_hidden_states[idx],
                        torch.zeros(max_k_len - len(all_encoder_hidden_states[idx]), C, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                    ], dim=0) if len(all_encoder_hidden_states[idx]) < max_k_len else all_encoder_hidden_states[idx]
                    all_text_input_embeds[idx] = torch.cat([
                        all_text_input_embeds[idx],
                        torch.zeros(max_q_len - len(all_text_input_embeds[idx]), C, device=text_input_embeds.device, dtype=text_input_embeds.dtype)
                    ], dim=0) if len(all_text_input_embeds[idx]) < max_q_len else all_text_input_embeds[idx]
                    if attention_mask.dim() == 2:
                        all_encoder_attention_mask[idx] = torch.cat([
                            all_encoder_attention_mask[idx],
                            torch.zeros(max_k_len - len(all_encoder_attention_mask[idx]), device=encoder_attention_mask.device, dtype=encoder_attention_mask.dtype)
                        ], dim=0) if len(all_encoder_attention_mask[idx]) < max_k_len else all_encoder_attention_mask[idx]
                        all_text_attention_mask[idx] = torch.cat([
                            all_text_attention_mask[idx],
                            torch.zeros(max_q_len - len(all_text_attention_mask[idx]), device=text_attention_mask.device, dtype=text_attention_mask.dtype)
                        ], dim=0) if len(all_text_attention_mask[idx]) < max_q_len else all_text_attention_mask[idx]
                    elif attention_mask.dim() == 4:
                        padding_size = all_encoder_attention_mask[idx].shape
                        padding_size = (padding_size[0], max_q_len - padding_size[1], padding_size[2])
                        all_encoder_attention_mask[idx] = torch.cat([
                                all_encoder_attention_mask[idx],
                                torch.zeros(padding_size, device=encoder_attention_mask.device, dtype=encoder_attention_mask.dtype)
                            ], dim=1) if all_encoder_attention_mask[idx].shape[1] < max_k_len else all_encoder_attention_mask[idx]
                        padding_size = all_encoder_attention_mask[idx].shape
                        padding_size = (padding_size[0], padding_size[1], max_k_len - padding_size[2])
                        all_encoder_attention_mask[idx] = torch.cat([
                                all_encoder_attention_mask[idx],
                                torch.zeros(padding_size, device=encoder_attention_mask.device, dtype=encoder_attention_mask.dtype)
                            ], dim=2) if all_encoder_attention_mask[idx].shape[2] < max_k_len else all_encoder_attention_mask[idx]
                        
                        padding_size = all_text_attention_mask[idx].shape
                        padding_size = (padding_size[0], max_q_len - padding_size[1], padding_size[2])
                        all_text_attention_mask[idx] = torch.cat([
                                all_text_attention_mask[idx],
                                torch.zeros(padding_size, device=text_attention_mask.device, dtype=text_attention_mask.dtype)
                            ], dim=1) if all_text_attention_mask[idx].shape[1] < max_q_len else all_text_attention_mask[idx]
                        padding_size = all_text_attention_mask[idx].shape
                        padding_size = (padding_size[0], padding_size[1], max_q_len - padding_size[2])
                        all_text_attention_mask[idx] = torch.cat([
                                all_text_attention_mask[idx],
                                torch.zeros(padding_size, device=text_attention_mask.device, dtype=text_attention_mask.dtype)
                            ], dim=2) if all_text_attention_mask[idx].shape[2] < max_q_len else all_text_attention_mask[idx]
                    else:
                        raise NotImplementedError(f'attention_mask.dim()={attention_mask.dim()}')
                encoder_hidden_states = torch.stack(all_encoder_hidden_states, dim=0)
                input_embeds = torch.stack(all_text_input_embeds, dim=0)
                encoder_attention_mask = torch.stack(all_encoder_attention_mask, dim=0)
                attention_mask = torch.stack(all_text_attention_mask, dim=0)
                
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        if benchmark_efficiency:
            end = time.time()
            efficiency_metrics["total_prefill_time"] += end - start
            efficiency_metrics["vit_forward_time_per_image"] = (end - start) / pixel_values.shape[0]
        
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
