# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on transformers/src/transformers/models/llama/modeling_llama.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch InternLM2 model."""
import math
import queue
import threading
import warnings
from typing import List, Optional, Tuple, Union
import torch.distributed as dist

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast,
                                           SequenceClassifierOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (add_start_docstrings,
                                add_start_docstrings_to_model_forward, logging,
                                replace_return_docstrings)

from functools import partial
try:
    from transformers.generation.streamers import BaseStreamer
except:  # noqa # pylint: disable=bare-except
    BaseStreamer = None

from .configuration_internlm2 import InternLM2Config

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = 'InternLM2Config'

flash_attn_func, flash_attn_varlen_func = None, None
pad_input, index_first_axis, unpad_input = None, None, None
try:
    from flash_attn import flash_attn_func as _flash_attn_func
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis as _index_first_axis
    from flash_attn.bert_padding import pad_input as _pad_input
    from flash_attn.bert_padding import unpad_input as _unpad_input

    flash_attn_func, flash_attn_varlen_func = _flash_attn_func, _flash_attn_varlen_func
    pad_input, index_first_axis, unpad_input = _pad_input, _index_first_axis, _unpad_input
    has_flash_attn = True
except:
    has_flash_attn = False
    
try:
    from ring_flash_attn import zigzag_ring_flash_attn_varlen_func, zigzag_ring_flash_attn_func
    from ring_flash_attn import ring_flash_attn_varlen_func, ring_flash_attn_func
    has_ring_flash_attn = True
except:
    has_ring_flash_attn = False
    


def _import_flash_attn():
    global flash_attn_func, flash_attn_varlen_func
    global pad_input, index_first_axis, unpad_input
    try:
        from flash_attn import flash_attn_func as _flash_attn_func
        from flash_attn import \
            flash_attn_varlen_func as _flash_attn_varlen_func
        from flash_attn.bert_padding import \
            index_first_axis as _index_first_axis
        from flash_attn.bert_padding import pad_input as _pad_input
        from flash_attn.bert_padding import unpad_input as _unpad_input
        flash_attn_func, flash_attn_varlen_func = _flash_attn_func, _flash_attn_varlen_func
        pad_input, index_first_axis, unpad_input = _pad_input, _index_first_axis, _unpad_input
    except ImportError:
        raise ImportError('flash_attn is not installed.')


# # naive ring attention extraction
# def extract_local(value, rank, world_size, dim=1):
#     """Extract local tensor across the sequence dimension."""
#     value_chunks = value.chunk(world_size, dim=dim)
#     local_value = value_chunks[rank]
#     return local_value.to(value.device)

# # copied from V2PE
# def extract_local(value, rank, world_size, dim=1):
#     """Extract local tensor across the sequence dimension."""
#     value_chunks = value.chunk(2 * world_size, dim=dim)
#     local_value = torch.cat(
#         [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
#     )
#     return local_value.to(value.device)


# # for zigzag extraction
# def extract_local(value, cu_seqlens, rank, world_size, dim=1, padding_value=0):
#     local_values = []
#     value = value.transpose(0, dim)
#     padding_size = 0
#     for i in range(len(cu_seqlens) - 1):
#         start, end = cu_seqlens[i], cu_seqlens[i + 1]
#         local_value = value[start:end].chunk(2*world_size, dim=0)
#         assert len(local_value[-1]) == len(local_value[0]), f"len(local_value[-1]): {len(local_value[-1])}, len(local_value[0]): {len(local_value[0])}"
#         local_values.extend(
#             [
#                 local_value[rank].detach().clone(),
#                 local_value[2 * world_size - 1 - rank].detach().clone(),
#             ]
#         )
#         if rank == 0:
#             # pad last chunk with zeros
#             chunk_size = local_value[0].shape[0]
#             padding_size += chunk_size - local_value[-1].shape[0]
#     padding_tensor = torch.zeros(padding_size, *local_values[-1].shape[1:], device=local_values[-1].device, dtype=local_values[-1].dtype)
#     if padding_value != 0:
#         padding_tensor.fill_(padding_value)
#     local_values.append(padding_tensor)
#     return torch.cat(local_values, dim=0).transpose(0, dim).contiguous()

def extract_local_idxs(value, cu_seqlens, rank, world_size, dim=1):
    local_values_idxs = []
    if isinstance(dim, int):
        dim = (dim,)
    for d in dim:
        assert value.shape[d] == value.shape[dim[0]], f"value.shape[d]: {value.shape[d]}, value.shape[dim[0]]: {value.shape[dim[0]]}"
    all_idxs = torch.arange(value.shape[dim[0]], device=value.device)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        local_idxs = all_idxs[start:end].chunk(2*world_size, dim=0)
        assert len(local_idxs[-1]) == len(local_idxs[0]), f"len(local_idxs[-1]): {len(local_idxs[-1])}, len(local_idxs[0]): {len(local_idxs[0])}"
        local_values_idxs.extend(
            [
                local_idxs[rank].detach().clone(),
                local_idxs[2 * world_size - 1 - rank].detach().clone(),
            ]
        )
    return torch.cat(local_values_idxs, dim=0)

def extract_local1(value, cu_seqlens, rank, world_size, dim=1, padding_value=0):
    if isinstance(dim, int):
        dim = (dim,)
    if isinstance(cu_seqlens[0], int):
        local_idxs = extract_local_idxs(value, cu_seqlens, rank, world_size, dim)
        for d in dim:
            value = value.index_select(d, local_idxs)
        return value
    else:
        all_local_idxs = []
        assert len(cu_seqlens) == len(dim), f"len(cu_seqlens): {len(cu_seqlens)}, len(dim): {len(dim)}"
        for i in range(len(cu_seqlens)):
            local_idxs = extract_local_idxs(value, cu_seqlens[i], rank, world_size, dim[i])
            all_local_idxs.append(local_idxs)
        for d, local_idxs in zip(dim, all_local_idxs):
            value = value.index_select(d, local_idxs)
        return value

def batch_extract_local(value, cu_seqlens, rank, world_size, dim=1, padding_value=0):
    """
    Args:
        value: [bsz, ...] tensor
        cu_seq_lens: list[list], len should be of batch size, each list contains the cumulative sequence lengths of each sequence in the batch
    """
    batch_local_values = []
    for i in range(value.shape[0]):
        local_values = extract_local1(value[i].unsqueeze(0), cu_seqlens[i], rank, world_size, dim, padding_value)
        batch_local_values.append(local_values)
    if isinstance(dim, int):
        dim = [dim]
    dim.sort()
    
    if len(dim) == 1:
        return torch.cat(batch_local_values, dim=dim[0])
    else:
        value_shape = list(batch_local_values[0].shape)
        dim_cumsum = []
        for d in dim:
            value_shape[d] = sum([x.shape[d] for x in batch_local_values])
            dim_cumsum.append([0] + torch.cumsum(torch.tensor([x.shape[d] for x in batch_local_values]), dim=0).tolist())
            
        return_value = torch.zeros(value_shape, device=value.device, dtype=value.dtype, requires_grad=value.requires_grad)
        for i, local_values in enumerate(batch_local_values):
            slice_value = return_value
            for k, d in enumerate(dim):
                start = dim_cumsum[k][i]
                end = dim_cumsum[k][i + 1]
                slices = [slice(None)] * len(value_shape)
                slices[d] = slice(start, end)
                slice_value = slice_value[tuple(slices)]
            slice_value[:] = local_values
            start = end
        return return_value
extract_local = batch_extract_local

def extract_local2(value, rank, world_size,  dim=1):
    """Extract local tensor across the hidden dimension."""
    dimension_size = value.shape[dim]
    sub_seq_length = dimension_size // world_size

    sub_seq_start = rank * sub_seq_length
    sub_seq_end = (rank + 1) * sub_seq_length
    local_value = value[:, sub_seq_start:sub_seq_end]

    return local_value.to(value.device)
# Modified from transformers.model.llama.modeling_llama.LlamaModel
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input, shape_list=None, dim=None):
        ctx.save_for_backward(input, shape_list)
        ctx.dim = dim
        if shape_list is None:
            output = [torch.zeros_like(input) for _ in range(dist.get_world_size(local_group))]
        else:
            output = [torch.zeros(torch.Size(shape), device=input.device, dtype=input.dtype) for shape in shape_list]
        # print(f"output: {output}", f"input: {input}", f"local_group: {local_group}")
        dist.all_gather(output, input, group=local_group)
        if dim is not None:
            return torch.cat(output, dim)
        else:
            return torch.stack(output)

    @staticmethod
    def backward(ctx, grads):
        (input, shape_list) = ctx.saved_tensors
        dim = ctx.dim
        
        dist.all_reduce(grads, group=local_group)
        grad_out = torch.zeros_like(input)
        
        if dim is None:
            grad_out[:] = grads[dist.get_rank(local_group)]
        else:
            if shape_list is None:
                input_shape = input.shape
                # grads are cat'ed along dim, so we need to split them
                grad_out[:] = grads.split(input_shape[dim], dim)[dist.get_rank(local_group)]
            else:
                seq_lens = [shape[dim] for shape in shape_list]
                grad_out[:] = grads.split(seq_lens, dim)[dist.get_rank(local_group)]
        return grad_out, None, None

# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    # attention_mask: [bsz, q_len] or [bsz, kv_len]
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
    
def find_next_example(mask_slice):
    """Given a matrix M (m, n), find the point (i, j) where M[:i][:j] all zero and M[:i][j:] all zero
        assuming M[:i][:j] is either 
        - (casual mask) left-bottom 1s and right-top 0s
        - (no mask) full 1s
    """
    i = mask_slice[:, 0].nonzero(as_tuple=False).flatten()
    if i.numel() == 0:
        # mask slice is the padding part
        return None, None
    i = i.max().item()
    j = mask_slice[i].nonzero(as_tuple=False).flatten()
    if j.numel() == 0:
        # mask slice is the padding part
        return None, None
    j = j.max().item()
    i += 1
    j += 1
    return i, j

def _get_unpad_packing_data(attention_mask, return_seq_len=False):
    """
    Process packed attention masks to get indices and cumulative sequence lengths.
    
    Args:
        attention_mask: [bsz, 1, q_len, kv_len] tensor containing packed attention masks
        
    Returns:
        tuple containing:
        - indices: flattened indices of non-zero elements
        - cu_seqlens: cumulative sequence lengths
        - max_seqlen_in_batch: maximum sequence length in the batch
    """
    
    indices = torch.nonzero(attention_mask.sum(-2).flatten(), as_tuple=False).flatten()
    mask = attention_mask.squeeze(1)  # [bsz, q_len, kv_len]
    bsz, q_len, kv_len = mask.shape
    
    seqlens = []
    
    # Process each batch
    for b in range(bsz):
        mask_slice = mask[b]
        
        sub_mask_idx = [0, 0]
        seqlens.append([])
        while sub_mask_idx[1] < kv_len:
            sub_mask_slice = mask_slice[sub_mask_idx[0]:, sub_mask_idx[1]:]
            q_i, kv_j = find_next_example(sub_mask_slice)
            if any(x is None for x in (q_i, kv_j)):
                # No more examples in this batch
                break
            seq_len = kv_j
                
            seqlens[-1].append(seq_len)
            sub_mask_idx[0] += q_i
            sub_mask_idx[1] += kv_j
        
    seqlens_tensor = torch.cat([torch.tensor(x, device=attention_mask.device, dtype=torch.int32) for x in seqlens])
    
    # Calculate cumulative sequence lengths
    cu_seqlens = F.pad(torch.cumsum(seqlens_tensor, dim=0, dtype=torch.int32), (1, 0))
    
    # Get maximum sequence length
    max_seqlen_in_batch = max([max(x) if x else 0 for x in seqlens]) if seqlens else 0
    
    if any([len(x) == 0 for x in seqlens]):
        print("Warning #### Some sequences have zero length.")
        raise ValueError("Some sequences have zero length.")
    if return_seq_len:
        return indices, cu_seqlens, max_seqlen_in_batch, seqlens
    return indices, cu_seqlens, max_seqlen_in_batch

def _get_unpad_packing_data_for_ct(encoder_attention_mask, cu_seqlens_q, seqlens_q):
    """
    Process packed attention masks to get indices and cumulative sequence lengths.
    
    Args:
        encoder_attention_mask: [bsz, 1, q_len, kv_len] tensor containing packed attention masks (cross-attention)
        attention_mask: [bsz, 1, q_len, q_len] tensor containing packed attention masks (self-attention)
    Returns:
        tuple containing:
        - indices: flattened indices of non-zero elements
        - cu_seqlens: cumulative sequence lengths
        - max_seqlen_in_batch: maximum sequence length in the batch
    """
    # indices = torch.nonzero(encoder_attention_mask.sum(-2).flatten(), as_tuple=False).flatten()
    indices = []
    mask = encoder_attention_mask.squeeze(1)  # [bsz, q_len, kv_len]
    bsz, q_len, kv_len = mask.shape
    
    seqlens = []
    cur_cu_q_idx = 0
    
    previous_q_seqlens = []
    # Process each batch
    for b in range(bsz):
        mask_slice = mask[b]
        
        sub_mask_idx = [0, 0]
        num_q_current_batch = 0
        while sub_mask_idx[1] < kv_len:
            sub_mask_slice = mask_slice[sub_mask_idx[0]:, sub_mask_idx[1]:]
            q_i, kv_j = find_next_example(sub_mask_slice)
            if any(x is None for x in (q_i, kv_j)):
                # No more examples in this batch
                # pad 0 if there are still some q sequences left
                if num_q_current_batch < len(seqlens_q[b]):
                    q_seq_lens = []
                    previous_seq_len_sum = sum(previous_q_seqlens)
                    while num_q_current_batch < len(seqlens_q[b]):
                        cur_cu_q_idx += 1
                        q_seq_lens.append(cu_seqlens_q[cur_cu_q_idx] - cu_seqlens_q[cur_cu_q_idx - 1])
                        assert q_seq_lens[-1] == seqlens_q[b][num_q_current_batch], f"q_seq_lens[-1]: {q_seq_lens[-1]}, seqlens_q[b][num_q_current_batch]: {seqlens_q[b][num_q_current_batch]}"
                        num_q_current_batch += 1
                    previous_q_seqlens.extend(q_seq_lens)
                    seqlens.extend([0] * len(q_seq_lens))
                break
            seq_len = kv_j
            
            # get the number q sequences between sub_mask_idx[0] and sub_mask_idx[0] + q_i
            q_seq_lens = []
            previous_seq_len_sum = sum(previous_q_seqlens)
            while cu_seqlens_q[cur_cu_q_idx] < previous_seq_len_sum + q_i:
                cur_cu_q_idx += 1
                q_seq_lens.append(cu_seqlens_q[cur_cu_q_idx] - cu_seqlens_q[cur_cu_q_idx - 1])
            assert sum(q_seq_lens) == q_i, f"q_seq_lens: {q_seq_lens}, q_i: {q_i}"
            previous_q_seqlens.extend(q_seq_lens)
            num_q_current_batch += len(q_seq_lens)
            
            seqlens.extend([seq_len] * len(q_seq_lens))
            indices.extend([torch.arange(b * kv_len + sub_mask_idx[1], b * kv_len + sub_mask_idx[1] + seq_len)] * len(q_seq_lens))
            
            sub_mask_idx[0] += q_i
            sub_mask_idx[1] += kv_j
            
        if num_q_current_batch < len(seqlens_q[b]):
            q_seq_lens = []
            previous_seq_len_sum = sum(previous_q_seqlens)
            while num_q_current_batch < len(seqlens_q[b]):
                cur_cu_q_idx += 1
                q_seq_lens.append(cu_seqlens_q[cur_cu_q_idx] - cu_seqlens_q[cur_cu_q_idx - 1])
                assert q_seq_lens[-1] == seqlens_q[b][num_q_current_batch], f"q_seq_lens[-1]: {q_seq_lens[-1]}, seqlens_q[b][num_q_current_batch]: {seqlens_q[b][num_q_current_batch]}"
                num_q_current_batch += 1
            previous_q_seqlens.extend(q_seq_lens)
            seqlens.extend([0] * len(q_seq_lens))
                    
    assert len(seqlens) == len(cu_seqlens_q) - 1, f"seqlens: {seqlens}, cu_seqlens_q: {cu_seqlens_q}"
    
    seqlens_tensor = torch.tensor(seqlens, device=encoder_attention_mask.device, dtype=torch.int32).flatten()
    
    # Calculate cumulative sequence lengths
    cu_seqlens = F.pad(torch.cumsum(seqlens_tensor, dim=0, dtype=torch.int32), (1, 0))
    
    indices = torch.cat(indices).to(encoder_attention_mask.device)
    
    # Get maximum sequence length
    max_seqlen_in_batch = max(seqlens) if seqlens else 0

    return indices, cu_seqlens, max_seqlen_in_batch

def unpad_packing_input(hidden_states, attention_mask, return_seq_len=False):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask_in_length: (batch, 1, q_len, kv_len), int, a nonzero number (e.g., 1, 2, 3, etc.) means length of concatenated sequence in b-th batch, and 0 means none.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    if return_seq_len:
        indices, cu_seqlens, max_seqlen_in_batch, seqlens = _get_unpad_packing_data(attention_mask, return_seq_len=return_seq_len)
        # Rearrange and index hidden states
        return (
            index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
            seqlens
        )
    else:
        indices, cu_seqlens, max_seqlen_in_batch = _get_unpad_packing_data(attention_mask)
        # Rearrange and index hidden states
        return (
            index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
        )
    
# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->InternLM2
class InternLM2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        InternLM2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.model.llama.modeling_llama.LlamaRotaryEmbedding with Llama->InternLM2
class InternLM2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device).to(dtype=self.inv_freq.dtype)

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=torch.float32)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.model.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->InternLM2
class InternLM2LinearScalingRotaryEmbedding(InternLM2RotaryEmbedding):
    """InternLM2RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device).to(dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)


# Copied from transformers.model.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->InternLM2
class InternLM2DynamicNTKScalingRotaryEmbedding(InternLM2RotaryEmbedding):
    """InternLM2RotaryEmbedding extended with Dynamic NTK scaling.
    Credits to the Reddit users /u/bloc97 and /u/emozilla.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer('inv_freq', inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device).to(dtype=self.inv_freq.dtype)

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)


# Copied from transformers.model.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.model.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_ct(q_or_k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors for cross attention."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_or_k_embed = (q_or_k * cos) + (rotate_half(q_or_k) * sin)
    return q_or_k_embed


class InternLM2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.w2(self.act_fn(self.w1(x)) * self.w3(x))

        return down_proj


# Copied from transformers.model.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Modified from transformers.model.llama.modeling_llama.LlamaAttention
class InternLM2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}'
                f' and `num_heads`: {self.num_heads}).'
            )

        self.wqkv = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.bias,
        )

        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = InternLM2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling['type']
            scaling_factor = self.config.rope_scaling['factor']
            if scaling_type == 'dynamic':
                self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == 'linear':
                self.rotary_emb = InternLM2LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError("Currently we only support rotary embedding's type being 'dynamic' or 'linear'.")
        return self.rotary_emb

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`'
            )

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.wqkv(hidden_states)

        qkv_states = rearrange(
            qkv_states,
            'b q (h gs d) -> b q h gs d',
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is'
                f' {attn_weights.size()}'
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}'
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
                f' {attn_output.size()}'
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class InternLM2CrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = False

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}'
                f' and `num_heads`: {self.num_heads}).'
            )

        self.wqkv = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.bias,
        )
        self.wq_kv_params = None

        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = InternLM2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling['type']
            scaling_factor = self.config.rope_scaling['factor']
            if scaling_type == 'dynamic':
                self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == 'linear':
                self.rotary_emb = InternLM2LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError("Currently we only support rotary embedding's type being 'dynamic' or 'linear'.")
        return self.rotary_emb

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def get_q_kv_weights(self):
        if not self.wq_kv_params:
            wqkv_weight = rearrange(
                self.wqkv.weight,
                '(h gs d) in -> h gs d in',
                gs=2 + self.num_key_value_groups,
                d=self.head_dim,
            )
            q_weights = wqkv_weight[:, : self.num_key_value_groups, :, :]
            kv_weights = wqkv_weight[:, self.num_key_value_groups:, :, :]
            q_weights = q_weights.reshape(-1, q_weights.size(-1))
            kv_weights = kv_weights.reshape(-1, kv_weights.size(-1))
            
            # for bias
            if self.wqkv.bias is not None:
                wqkv_bias = rearrange(
                    self.wqkv.bias,
                    '(h gs d) -> h gs d',
                    gs=2 + self.num_key_value_groups,
                    d=self.head_dim,
                )
                q_bias = wqkv_bias[:, : self.num_key_value_groups, :]
                kv_bias = wqkv_bias[:, self.num_key_value_groups:, :]
                q_bias = q_bias.flatten()
                kv_bias = kv_bias.flatten()
            else:
                q_bias=None
                kv_bias=None
            self.wq_kv_params = (q_weights, kv_weights, q_bias, kv_bias)
        else:
            q_weights, kv_weights, q_bias, kv_bias = self.wq_kv_params
        return q_weights, kv_weights, q_bias, kv_bias
            
    @staticmethod
    def _forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`'
            )

        bsz, q_len, _ = hidden_states.size()
        
        encoder_bsz, encoder_q_len, _ = encoder_hidden_states.size()
        assert bsz == encoder_bsz, f"Batch size of query and key must be the same. Got {bsz} and {encoder_bsz}"
        
        q_w, kv_w, q_bias, kv_bias = self.get_q_kv_weights()
        
        query_states = F.linear(hidden_states, q_w, q_bias)
        kv_states = F.linear(encoder_hidden_states, kv_w, kv_bias)
        query_states = rearrange(query_states, 'b q (h d) -> b q h d', d=self.head_dim)
        kv_states = rearrange(kv_states, 'b q (h gs d) -> b q h gs d', gs=2, d=self.head_dim)
        key_states = kv_states[..., -2, :]
        value_states = kv_states[..., -1, :]
        
        # ## Original
        # qkv_states = self.wqkv(hidden_states)

        # qkv_states = rearrange(
        #     qkv_states,
        #     'b q (h gs d) -> b q h gs d',
        #     gs=2 + self.num_key_value_groups,
        #     d=self.head_dim,
        # )

        # query_states = qkv_states[..., : self.num_key_value_groups, :]
        # query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
        # key_states = qkv_states[..., -2, :]
        # value_states = qkv_states[..., -1, :]
        # ## End Original

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        q_len = query_states.shape[-2]
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, q_len))
        query_states = apply_rotary_pos_emb_ct(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_ct(key_states, cos, sin, encoder_position_ids)
        
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is'
                f' {attn_weights.size()}'
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}'
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
                f' {attn_output.size()}'
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    def forward(self, *args, **kwargs):
        return self._forward(self, *args, **kwargs)

class InternLM2SDPAAttention(InternLM2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # InternLM2FlashAttention2 attention does not support output_attentions
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`'
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop('padding_mask')

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.wqkv(hidden_states)

        qkv_states = rearrange(
            qkv_states,
            'b q (h gs d) -> b q h gs d',
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max() + 1)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}'
                )
        
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attention_mask.bool(), enable_gqa=True
        )
        
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
class InternLM2SDPACrossAttention(InternLM2CrossAttention):
    """
    InternLM2 flash attention module. This module inherits from `InternLM2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    @staticmethod
    def _forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # InternLM2FlashAttention2 attention does not support output_attentions
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`'
            )

            # overwrite attention_mask with padding_mask
            # attention_mask = kwargs.pop('padding_mask')
            kwargs.pop('padding_mask')

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        
        encoder_bsz, encoder_q_len, _ = encoder_hidden_states.size()
        assert bsz == encoder_bsz, f"Batch size of query and key must be the same. Got {bsz} and {encoder_bsz}"
        
        q_w, kv_w, q_bias, kv_bias = self.get_q_kv_weights()
        
        query_states = F.linear(hidden_states, q_w, q_bias)
        kv_states = F.linear(encoder_hidden_states, kv_w, kv_bias)
        query_states = rearrange(query_states, 'b q (h d) -> b q h d', d=self.head_dim)
        kv_states = rearrange(kv_states, 'b q (h gs d) -> b q h gs d', gs=2, d=self.head_dim)
        key_states = kv_states[..., -2, :]
        value_states = kv_states[..., -1, :]
        
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        q_len = query_states.shape[-2]
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=max(position_ids.max() + 1, encoder_position_ids.max() + 1))
        query_states = apply_rotary_pos_emb_ct(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_ct(key_states, cos, sin, encoder_position_ids)
        
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            
        past_key_value = (key_states, value_states) if use_cache else None
        

        if encoder_attention_mask is not None:
            if encoder_attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {encoder_attention_mask.size()}'
                )
        # indices_q, cu_seq_lens_q, max_seq_lens_q = _get_unpad_packing_data(attention_mask)
        # indices_k, cu_seq_lens_k, max_seq_lens_k = _get_unpad_packing_data(encoder_attention_mask)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, encoder_attention_mask.bool(), enable_gqa=True
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def forward(self, *args, **kwargs):
        return self._forward(self, *args, **kwargs)
    
# Modified from transformers.model.llama.modeling_llama.InternLM2FlashAttention2
class InternLM2FlashAttention2(InternLM2Attention):
    """
    InternLM2 flash attention module. This module inherits from `InternLM2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """
    
    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        self.class_flash_attn_varlen_func = flash_attn_varlen_func
        self.class_flash_attn_func = flash_attn_func
        self.use_ring_attn = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # InternLM2FlashAttention2 attention does not support output_attentions
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`'
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop('padding_mask')

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.wqkv(hidden_states)

        qkv_states = rearrange(
            qkv_states,
            'b q (h gs d) -> b q h gs d',
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        
        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        causal = self.is_causal and query_length != 1
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            if attention_mask is not None and attention_mask.dim() != 2:
                assert attention_mask.dim() == 4, f"Attention mask should be 4D for packing input, got {attention_mask.dim()}"
                query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._unpad_packing_input(
                    query_states, key_states, value_states, attention_mask, query_length
                )
            else:
                query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._unpad_input(
                    query_states, key_states, value_states, attention_mask, query_length
                )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if self.use_ring_attn:
                assert (cu_seqlens_q == cu_seqlens_k).all(), f"{cu_seqlens_q} != {cu_seqlens_k}"
                assert (max_seqlen_in_batch_q == max_seqlen_in_batch_k), f"{max_seqlen_in_batch_q} != {max_seqlen_in_batch_k}"
                world_size = dist.get_world_size(local_group)
                assert all([x % (2) == 0 for x in cu_seqlens_q]), f"{cu_seqlens_q} % {2 * world_size} != 0"
                assert all([x % (2) == 0 for x in cu_seqlens_k]), f"{cu_seqlens_k} % {2 * world_size} != 0"
                attn_output_unpad = self.class_flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens=cu_seqlens_q,
                    max_seqlen=max_seqlen_in_batch_q,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = self.class_flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = self.class_flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output
    
    def _unpad_packing_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        bsz, q_len, _, _ = query_layer.shape
        bsz, kv_seq_len, _, _ = key_layer.shape
        # packing attention mask should be 4d (batch_size, num_heads, query_length, key_length)
        if attention_mask.dim() == 4 and attention_mask.size() == (bsz, 1, q_len, kv_seq_len):
            pass
        elif attention_mask.dim() == 3 and attention_mask.size() == (bsz, q_len, kv_seq_len):
            pass
        else:
            raise ValueError(
                f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)} or {(bsz, q_len, kv_seq_len)}, but is {attention_mask.size()}'
            )
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_packing_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            raise ValueError("Packing self-attention does not support query_length != kv_seq_len")
            # The -q_len: slice assumes left padding.
            # attention_mask = attention_mask[:, -query_length:]
            # query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q.to(torch.int64),
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
        
    def _unpad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q.to(torch.int64),
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
        
class InternLM2FlashCrossAttention2(InternLM2CrossAttention):
    """
    InternLM2 flash attention module. This module inherits from `InternLM2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """
    
    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        self.class_flash_attn_varlen_func = flash_attn_varlen_func
        self.class_flash_attn_func = flash_attn_func

    @staticmethod
    def _forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # InternLM2FlashAttention2 attention does not support output_attentions
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`'
            )

            # overwrite attention_mask with padding_mask
            # attention_mask = kwargs.pop('padding_mask')
            kwargs.pop('padding_mask')

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        
        encoder_bsz, encoder_q_len, _ = encoder_hidden_states.size()
        assert bsz == encoder_bsz, f"Batch size of query and key must be the same. Got {bsz} and {encoder_bsz}"
        
        q_w, kv_w, q_bias, kv_bias = self.get_q_kv_weights()
        
        query_states = F.linear(hidden_states, q_w, q_bias)
        kv_states = F.linear(encoder_hidden_states, kv_w, kv_bias)
        query_states = rearrange(query_states, 'b q (h d) -> b q h d', d=self.head_dim)
        kv_states = rearrange(kv_states, 'b q (h gs d) -> b q h gs d', gs=2, d=self.head_dim)
        key_states = kv_states[..., -2, :]
        value_states = kv_states[..., -1, :]
        
        # ## Original
        # qkv_states = self.wqkv(hidden_states)

        # qkv_states = rearrange(
        #     qkv_states,
        #     'b q (h gs d) -> b q h gs d',
        #     gs=2 + self.num_key_value_groups,
        #     d=self.head_dim,
        # )

        # _query_states = qkv_states[..., : self.num_key_value_groups, :]
        # _query_states = rearrange(_query_states, 'b q h gs d -> b q (h gs) d')
        # _key_states = qkv_states[..., -2, :]
        # _value_states = qkv_states[..., -1, :]
        
        # # compute difference
        # for key, value, _value in zip(['query', 'key', 'value'], [query_states, key_states, value_states], [_query_states, _key_states, _value_states]):
        #     print(f"{key} diff: {(value - _value).abs().mean()}")
        # ## End Original

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        q_len = query_states.shape[-2]
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        cos, sin = self.rotary_emb(value_states, seq_len=max(position_ids.max().item() + 1, encoder_position_ids.max().item() + 1))
        query_states = apply_rotary_pos_emb_ct(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_ct(key_states, cos, sin, encoder_position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, encoder_attention_mask, q_len
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def forward(self, *args, **kwargs):
        return self._forward(self, *args, **kwargs)

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, encoder_attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        causal = self.is_causal and query_length != 1
        if attention_mask is not None or encoder_attention_mask is not None:
            batch_size = query_states.shape[0]
            if attention_mask is not None and attention_mask.dim() != 2:
                query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._unpad_packing_input(
                    query_states, key_states, value_states, attention_mask, encoder_attention_mask, query_length
                )
            else:
                query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._unpad_input(
                    query_states, key_states, value_states, attention_mask, encoder_attention_mask, query_length
                )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            
            attn_output_unpad = self.class_flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = self.class_flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _unpad_packing_input(self, query_layer, key_layer, value_layer, attention_mask, encoder_attention_mask, query_length):
        # packing attention mask should be 4d (batch_size, num_heads, query_length, key_length)
        bsz, q_len, _, _ = query_layer.size()
        bsz, kv_seq_len, _, _ = key_layer.size()
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones((bsz, 1, q_len, kv_seq_len), device=key_layer.device)
        if encoder_attention_mask.dim() == 4 and encoder_attention_mask.size() == (bsz, 1, q_len, kv_seq_len):
            pass
        elif encoder_attention_mask.dim() == 3 and encoder_attention_mask.size() == (bsz, q_len, kv_seq_len):
            pass
        else:
            raise ValueError(
                f'Encoder attention mask should be of size {(bsz, 1, q_len, kv_seq_len)} or {(bsz, q_len, kv_seq_len)}, but is {encoder_attention_mask.size()}'
            )
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, 1, query_length, query_length), device=query_layer.device)
        # The -q_len: slice assumes left padding.
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, seqlens_q = unpad_packing_input(query_layer, attention_mask, return_seq_len=True)
        
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_packing_data_for_ct(encoder_attention_mask, cu_seqlens_q, seqlens_q)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
            
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q.to(torch.int64),
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
        
    def _unpad_input(self, query_layer, key_layer, value_layer, attention_mask, encoder_attention_mask, query_length):
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones((batch_size, kv_seq_len), device=key_layer.device)
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(encoder_attention_mask)

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, query_length), device=query_layer.device)
            # The -q_len: slice assumes left padding.
            # attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)[:4]

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q.to(torch.int64),
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

class InternLM2RingFlashAttention2(InternLM2FlashAttention2):
    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        self.class_flash_attn_varlen_func = zigzag_ring_flash_attn_varlen_func
        self.class_flash_attn_func = zigzag_ring_flash_attn_func
        self.use_ring_attn = True
class InternLM2RingFlashCrossAttention2(InternLM2FlashCrossAttention2):
    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        self.class_flash_attn_varlen_func = ring_flash_attn_varlen_func
        self.class_flash_attn_func = ring_flash_attn_func
        self.use_ring_attn = True
    
    
INTERNLM2_ATTENTION_CLASSES = {
    'eager': InternLM2Attention,
    'flash_attention_2': InternLM2FlashAttention2,
    'sdpa': InternLM2SDPAAttention,
    'ring_flash_attn': InternLM2FlashAttention2,
}

INTERNLM2_CROSS_ATTENTION_CLASSES = {
    'eager': InternLM2CrossAttention,
    'flash_attention_2': InternLM2FlashCrossAttention2,
    'sdpa': InternLM2SDPACrossAttention,
    'ring_flash_attn': InternLM2RingFlashCrossAttention2,
}

# Modified from transformers.model.llama.modeling_llama.LlamaDecoderLayer
class InternLM2DecoderLayer(nn.Module):
    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.enable_cross_attention = config.enable_cross_attention
        self.enable_shared_cross_attention = config.enable_shared_cross_attention
        self.local_attention_group_size = config.local_attention_group_size

        if self.enable_cross_attention:
            self.attention = INTERNLM2_ATTENTION_CLASSES[config.attn_implementation](config=config)
            self.cross_attention = INTERNLM2_CROSS_ATTENTION_CLASSES[config.attn_implementation](config=config)
            self.cross_attention_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1))
        elif self.enable_shared_cross_attention:
            # use cross attention for both self and cross attention
            self.attention = INTERNLM2_CROSS_ATTENTION_CLASSES[config.attn_implementation](config=config)
            self.attention.is_causal = True
        else:
            # use cross attention for both self and cross attention
            self.attention = INTERNLM2_ATTENTION_CLASSES[config.attn_implementation](config=config)
            
        
            
        self.feed_forward = InternLM2MLP(config)
        self.attention_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            
        self.ffn_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor=None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`'
            )
        
        output_encoder_hidden_states = False
        if encoder_hidden_states is None or (not self.enable_cross_attention and not self.enable_shared_cross_attention):
            residual = hidden_states

            hidden_states = self.attention_norm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = residual + hidden_states
        # Cross Attention
        elif encoder_hidden_states is not None:
            if self.enable_cross_attention:
                residual = hidden_states

                hidden_states = self.attention_norm(hidden_states)

                # Self Attention
                hidden_states, self_attn_weights, present_key_value = self.attention(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
                )
                hidden_states = residual + hidden_states
                
                residual = hidden_states
                
                hidden_states = self.cross_attention_norm(hidden_states)
                hidden_states, cross_attn_weights, _ = self.cross_attention(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    position_ids=position_ids,
                    encoder_position_ids=encoder_position_ids,
                    output_attentions=output_attentions,
                    **kwargs,
                )
                hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states
            elif self.enable_shared_cross_attention:
                # first norm
                output_encoder_hidden_states = True
                residual = hidden_states
                hidden_states = self.attention_norm(hidden_states)
                bsz, kv_seq_len, _ = encoder_hidden_states.size()
                # first self attention using hidden_states as the query, and encoder_hidden_states as the key and value
                if not use_cache or past_key_value is None:
                    # we don't use the original encoder_hidden_states here as it keeps to be the orignal one without self attention
                    # instead, we extract the encoder_hidden_states from the hidden_states of size [bsz, kv_seq_len+q_len, hidden_size]
                    residual_encoder = encoder_hidden_states
                    encoder_hidden_states = self.attention_norm(encoder_hidden_states)
                    merged_kv_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                    merged_kv_position_ids = torch.cat([encoder_position_ids, position_ids], dim=1)
                else:
                    merged_kv_hidden_states = hidden_states
                    merged_kv_position_ids = position_ids
                
                # then cross attention
                hidden_states, self_attn_weights, present_key_value = self.attention(
                    hidden_states=hidden_states,
                    encoder_hidden_states=merged_kv_hidden_states,
                    attention_mask=attention_mask,
                    encoder_attention_mask=attention_mask,
                    position_ids=position_ids,
                    encoder_position_ids=merged_kv_position_ids,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    past_key_value=past_key_value,
                )
                
                # locally self attention for the encoder_hidden_states
                if not use_cache or past_key_value is None:
                    # local self attention for cross attention
                    kv_seq_len = encoder_hidden_states.size(1)
                    chunk_idxs = torch.arange(1, kv_seq_len, device=hidden_states.device) # 0 is bos token
                    chunk_idxs = torch.split(chunk_idxs, self.local_attention_group_size)
                    
                    local_self_attn_output = []
                    encoder_local_attention_mask = encoder_attention_mask
                    previous_sparse_attn_idxs = [0]
                    for i, local_idxs in enumerate(chunk_idxs):
                        local_idxs = torch.cat([torch.tensor(previous_sparse_attn_idxs, device=hidden_states.device), local_idxs]) # add bos to the group for attending
                        local_encoder_hidden_states = encoder_hidden_states[:, local_idxs]
                        local_encoder_position_ids = encoder_position_ids[:, local_idxs]
                        if encoder_attention_mask is None:
                            local_encoder_attention_mask = None
                        elif encoder_local_attention_mask.dim() == 4:
                            local_encoder_attention_mask = encoder_local_attention_mask[:, :, local_idxs, :][:, :, :, local_idxs]
                        else:
                            local_encoder_attention_mask = encoder_local_attention_mask[:, local_idxs]
                        local_encoder_hidden_states, _, _ = self.attention(
                            hidden_states=local_encoder_hidden_states,
                            attention_mask=local_encoder_attention_mask,
                            encoder_hidden_states=local_encoder_hidden_states,
                            encoder_attention_mask=local_encoder_attention_mask,
                            position_ids=local_encoder_position_ids,
                            encoder_position_ids=local_encoder_position_ids,
                            past_key_value=None,
                            output_attentions=False,
                            use_cache=False,
                        )
                        if i == 0:
                            local_self_attn_output.append(local_encoder_hidden_states)
                        else:
                            local_self_attn_output.append(local_encoder_hidden_states[:, len(previous_sparse_attn_idxs):])
                        # previous_sparse_attn_idxs.extend(chunk_idxs[i].reshape(-1, 258)[:, [0, -1]].flatten().tolist()) # 258 is the magic number: 256 + 2. 256 is the num tokens per grid
                    encoder_hidden_states = torch.cat(local_self_attn_output, dim=1)
                
                    # # # DEBUG: compare difference, comment out this part when debugging
                    # _residual = self.attention_norm(torch.cat([residual_encoder, residual], dim=1))
                    # ori_hidden_states, _, _ = self.attention(
                    #     hidden_states=_residual,
                    #     encoder_hidden_states=_residual,
                    #     attention_mask=None,
                    #     encoder_attention_mask=None,
                    #     position_ids=torch.cat([encoder_position_ids, position_ids], dim=1),
                    #     encoder_position_ids=torch.cat([encoder_position_ids, position_ids], dim=1),
                    #     output_attentions=output_attentions,
                    #     use_cache=False,
                    #     past_key_value=None,
                    # )
                    # print(f"encoder part diff: {(encoder_hidden_states - ori_hidden_states[:, :kv_seq_len]).abs().mean()}")
                    # print(f"dense part diff: {(hidden_states - ori_hidden_states[:, kv_seq_len:]).abs().mean()}")
                    # # # DEBUG: compare difference, comment out this part when debugging
                    
                    encoder_hidden_states = residual_encoder + encoder_hidden_states
                    # ffn here for encoder_hidden_states
                    residual_encoder = encoder_hidden_states
                    encoder_hidden_states = self.ffn_norm(encoder_hidden_states)
                    encoder_hidden_states = self.feed_forward(encoder_hidden_states)
                    encoder_hidden_states = residual_encoder + encoder_hidden_states
                
                # add residual 
                hidden_states = residual + hidden_states
            else:
                raise ValueError("Cross attention is not enabled")
        else:
            raise ValueError("Cross attention is not enabled but encoder_hidden_states is not None")
                    
        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)
            
        if output_encoder_hidden_states:
            outputs += (encoder_hidden_states,)
        return outputs


InternLM2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`InternLM2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


# Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel with Llama->InternLM2
@add_start_docstrings(
    'The bare InternLM2 Model outputting raw hidden-states without any specific head on top.',
    InternLM2_START_DOCSTRING,
)
class InternLM2PreTrainedModel(PreTrainedModel):
    config_class = InternLM2Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternLM2DecoderLayer']
    _skip_keys_device_placement = 'past_key_values'
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


InternLM2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or
            when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# Modified from transformers.model.llama.modeling_llama.LlamaModel
@add_start_docstrings(
    'The bare InternLM2 Model outputting raw hidden-states without any specific head on top.',
    InternLM2_START_DOCSTRING,
)
class InternLM2Model(InternLM2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`InternLM2DecoderLayer`]

    Args:
        config: InternLM2Config
    """

    _auto_class = 'AutoModel'

    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        if not has_flash_attn:
            self.config.attn_implementation = 'eager'
            print('Warning: Flash attention is not available, using eager attention instead.')

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList([InternLM2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.tok_embeddings

    def set_input_embeddings(self, value):
        self.tok_embeddings = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        global local_group
        if hasattr(self, 'group_list') and self.group_list is not None:
            for group_idx,group in enumerate(self.group_list):
                if type(group)==torch.distributed.distributed_c10d.ProcessGroup:
                    break
            global inner_idx
            inner_idx = dist.get_rank(group)
            local_group=group
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.attn_implementation == 'flash_attention_2':
            _import_flash_attn()

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
        
        if encoder_position_ids is None and encoder_hidden_states is not None:
            encoder_position_ids = torch.arange(
                encoder_hidden_states.shape[1], dtype=torch.long, device=encoder_hidden_states.device
            )
            encoder_position_ids = encoder_position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)

        # prepare attention mask
        encoder_attention_mask_expaned_for_normal_attention = False
        if self.config.attn_implementation == 'flash_attention_2':
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            encoder_attention_mask = encoder_attention_mask if (encoder_attention_mask is not None and 0 in encoder_attention_mask) else None
        else:
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                )
            if encoder_hidden_states is not None:
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(
                        (batch_size, encoder_hidden_states.shape[1]), dtype=torch.bool, device=inputs_embeds.device
                    )
        
        encoder_local_attention_mask = None
        # prepare different attention masks if enable_shared_cross_attention is True
        if self.config.enable_shared_cross_attention and encoder_hidden_states is not None:
            # prepare encoder local attention mask, this is used for the local grouped self attention in the encoder kv states
            if encoder_attention_mask is None:
                encoder_local_attention_mask = None
            # get group len of each encoder hidden states
            elif encoder_attention_mask.dim() == 2:
                encoder_local_attention_mask = encoder_attention_mask
            else:
                if encoder_attention_mask_expaned_for_normal_attention:
                    _encoder_attention_mask = 1 - encoder_attention_mask
                else:
                    _encoder_attention_mask = encoder_attention_mask
                # [bsz, 1, q_len, kv_len]
                # encoder_attention_mask_min_q = _encoder_attention_mask.min(dim=-2).values
                encoder_attention_mask_max_q = _encoder_attention_mask.max(dim=-2).values
                # assert (((encoder_attention_mask_max_q[..., 1:] == encoder_attention_mask_max_q[..., :-1]) \
                    # == (encoder_attention_mask_min_q[..., 1:] == encoder_attention_mask_min_q[..., :-1])) \
                    # | (encoder_attention_mask_max_q[..., 1:] == 0)).all(), \
                    # 'Encoder attention mask should have the same group length'
                def create_causal_pattern_batched(x):
                    # Get batch size and sequence length
                    batch_size, seq_len = x.shape
                    
                    # Initialize result tensor with zeros
                    result = torch.zeros((batch_size, seq_len, seq_len), device=x.device)
                    
                    # Process each batch independently
                    for b in range(batch_size):
                        # Get unique values and their counts for this batch
                        unique, counts = torch.unique_consecutive(x[b], return_counts=True)
                        
                        # Keep track of starting position
                        start_idx = 0
                        
                        # For each unique value and its count
                        for val, count in zip(unique, counts):
                            # Create a causal pattern within this group
                            for i in range(count):
                                row = start_idx + i
                                # Fill only up to current position within the group
                                result[b, row, start_idx:start_idx + i + 1] = 1
                                
                            start_idx += count
                    return result
    
                encoder_local_attention_mask = create_causal_pattern_batched(encoder_attention_mask_max_q.squeeze(1).to(torch.int))
                encoder_local_attention_mask = encoder_local_attention_mask.unsqueeze(1)
                
            # then attention mask with be of size [bsz, 1, q_len, q_len+kv_len]
            # and encoder_attention_mask will be the encoder_local_attention_mask of size [bsz, 1, kv_len, kv_len] for local attention
            batch_size, q_len, _ = inputs_embeds.size()
            batch_size, kv_seq_len, _ = encoder_hidden_states.size()
            if encoder_attention_mask is None:
                if attention_mask is None:
                    merged_kv_attention_mask = None
                elif attention_mask.dim() == 4:
                    merged_kv_attention_mask = torch.cat([
                        torch.ones((batch_size, 1, q_len, kv_seq_len), device=attention_mask.device, dtype=attention_mask.dtype),
                        attention_mask
                    ], dim=3)
                else:
                    merged_kv_attention_mask = torch.cat([
                        torch.ones((batch_size, q_len), device=attention_mask.device, dtype=attention_mask.dtype),
                        attention_mask
                    ], dim=1)
            else:  
                if encoder_attention_mask.dim() == 4:
                    if attention_mask is None:
                        merged_kv_attention_mask = torch.cat([
                            encoder_attention_mask, 
                            torch.ones((batch_size, 1, q_len, q_len), device=encoder_attention_mask.device, dtype=encoder_attention_mask.dtype)
                        ], dim=3)
                    else:
                        merged_kv_attention_mask = torch.cat([encoder_attention_mask, attention_mask], dim=3)
                else:
                    # dim is 2
                    if attention_mask is None:
                        merged_kv_attention_mask = torch.cat([
                            encoder_attention_mask, 
                            torch.ones((batch_size, q_len), device=encoder_attention_mask.device, dtype=encoder_attention_mask.dtype)
                        ], dim=1)
                    else:  
                        merged_kv_attention_mask = torch.cat([encoder_attention_mask, attention_mask], dim=1)
            
            attention_mask = merged_kv_attention_mask # [bsz, 1, q_len, kv_len+q_len] or [bsz, kv_len+q_len]
            encoder_attention_mask = encoder_local_attention_mask # [bsz, 1, kv_len, kv_len] or [bsz, kv_len]
        
        # expand the attention mask if necessary    
        if not self.config.attn_implementation == 'flash_attention_2':
            if attention_mask.dim() == 2:
                attention_mask = self._prepare_decoder_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
            else:
                # attention_mask is already 3D(4D for attention heads)
                attention_mask = attention_mask.to(inputs_embeds.device)
            if encoder_attention_mask.dim() == 2:
                encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=seq_length).to(
                    inputs_embeds.device
                )
                encoder_attention_mask_expaned_for_normal_attention = True
            else:
                # encoder_attention_mask is already 3D(4D for attention heads)
                encoder_attention_mask = encoder_attention_mask.to(inputs_embeds.device)

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        if self.config.attn_implementation == 'ring_flash_attn':
            _, cu_seq_lens_q, _, seqlens = _get_unpad_packing_data(attention_mask, return_seq_len=True)
            batch_cu_seq_lens_q = [
                [0] + list(torch.cumsum(torch.tensor(x, dtype=torch.int32), dim=0).tolist())
                for x in seqlens
            ]
            if encoder_attention_mask is not None:
                _, cu_seq_lens_k, _, seqlens = _get_unpad_packing_data(encoder_attention_mask, return_seq_len=True)
                batch_cu_seq_lens_k = [
                    [0] + list(torch.cumsum(torch.tensor(x, dtype=torch.int32), dim=0).tolist())
                    for x in seqlens
                ]
            world_size = dist.get_world_size(local_group)
            rank = dist.get_rank(local_group)
            assert all([x % (2 * world_size) == 0 for y in batch_cu_seq_lens_q for x in y]), 'batch_cu_seq_lens_q should be divisible by 2 * world_size'
            assert all([x % (2 * world_size) == 0 for y in batch_cu_seq_lens_k for x in y]), 'batch_cu_seq_lens_k should be divisible by 2 * world_size'
            all_rank_hidden_states = []
            for _rank in range(world_size):
                all_rank_hidden_states.append(extract_local(hidden_states, batch_cu_seq_lens_q, _rank, world_size))
            assert all([x.shape == all_rank_hidden_states[0].shape for x in all_rank_hidden_states]), f'All rank hidden states should have the same shape.{[x.shape for x in all_rank_hidden_states]}'
            hidden_states = extract_local(hidden_states, batch_cu_seq_lens_q, rank, world_size)
            encoder_hidden_states = extract_local(encoder_hidden_states, batch_cu_seq_lens_k, rank, world_size) if encoder_hidden_states is not None else None
            
            if attention_mask.dim() == 2:
                attention_mask = extract_local(attention_mask, batch_cu_seq_lens_q, rank, world_size)
            else:
                assert attention_mask.dim() == 4, 'Attention mask should be 2D or 4D' # [bsz, 1, q_len, kv_len]
                attention_mask = extract_local(attention_mask, batch_cu_seq_lens_q, rank, world_size, dim=[-2, -1])
            if encoder_attention_mask is not None:
                if encoder_attention_mask.dim() == 2:
                    encoder_attention_mask = extract_local(encoder_attention_mask, batch_cu_seq_lens_k, rank, world_size)
                    encoder_local_attention_mask = extract_local(encoder_local_attention_mask, batch_cu_seq_lens_k, rank, world_size)
                else:
                    assert encoder_attention_mask.dim() == 4, 'Attention mask should be 2D or 4D'
                    batch_cu_seq_lens_q_k = [[x,y] for x,y in zip(batch_cu_seq_lens_q, batch_cu_seq_lens_k)]
                    encoder_attention_mask = extract_local(encoder_attention_mask, batch_cu_seq_lens_q_k, rank, world_size, dim=[-2, -1])
                    encoder_local_attention_mask = extract_local(encoder_local_attention_mask, batch_cu_seq_lens_q_k, rank, world_size, dim=[-2, -1])
            assert all(1 in attention_mask[i] for i in range(attention_mask.size(0))), 'Attention mask should have 1'
            
            position_ids = extract_local(position_ids, batch_cu_seq_lens_q, rank, world_size)
            encoder_position_ids = extract_local(encoder_position_ids, batch_cu_seq_lens_k, rank, world_size) if encoder_position_ids is not None else None
            
            _, local_cu_seq_lens_q, _ = _get_unpad_packing_data(attention_mask)
            if encoder_attention_mask is not None:
                _, local_cu_seq_lens_k, _ = _get_unpad_packing_data(encoder_attention_mask)
            world_size = dist.get_world_size(local_group)
            rank = dist.get_rank(local_group)
            assert all([x == (y // world_size) for x, y in zip(local_cu_seq_lens_q, cu_seq_lens_q)]), 'cu_seq_lens_q should be divisible by world_size'
            assert all([x == (y // world_size) for x, y in zip(local_cu_seq_lens_k, cu_seq_lens_k)]), 'cu_seq_lens_k should be divisible by world_size'
            
        else:
            pass

        for idx, decoder_layer in enumerate(self.layers):
            # print(f"Decoder layer {idx}")
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    encoder_hidden_states,
                    attention_mask,
                    encoder_attention_mask,
                    position_ids,
                    encoder_position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    position_ids=position_ids,
                    encoder_position_ids=encoder_position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            if encoder_hidden_states is not None and self.config.enable_shared_cross_attention:
                # update encoder_hidden_states
                encoder_hidden_states = layer_outputs[-1]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# Modified from transformers.model.llama.modeling_llama.LlamaForCausalLM
class InternLM2ForCausalLM(InternLM2PreTrainedModel):
    _auto_class = 'AutoModelForCausalLM'

    _tied_weights_keys = ['output.weight']

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.output

    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, InternLM2ForCausalLM

        >>> model = InternLM2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        global local_group
        if hasattr(self, 'group_list') and self.group_list is not None:
            for group_idx,group in enumerate(self.group_list):
                if type(group)==torch.distributed.distributed_c10d.ProcessGroup:
                    break
            global inner_idx
            inner_idx = dist.get_rank(group)
            local_group=group
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        
        if self.config.attn_implementation == 'ring_flash_attn':
            _, _, _, seqlens = _get_unpad_packing_data(attention_mask, return_seq_len=True)
            batch_cu_seq_lens_q = [
                [0] + list(torch.cumsum(torch.tensor(x, dtype=torch.int32), dim=0).tolist())
                for x in seqlens
            ]
            world_size = dist.get_world_size(local_group)
            rank = dist.get_rank(local_group)
            labels = extract_local(labels, batch_cu_seq_lens_q, rank, world_size, padding_value=-100) if labels is not None else None
        debug_ring_attention = False
        if debug_ring_attention:
            output_hidden_states = True
            self.config.attn_implementation = 'flash_attention_2'
            for layer in self.model.layers:
                layer.attention.class_flash_attn_varlen_func = flash_attn_varlen_func
                layer.attention.class_flash_attn_func = flash_attn_func
                layer.attention.use_ring_attn = False
                layer.cross_attention.class_flash_attn_varlen_func = flash_attn_varlen_func
                layer.cross_attention.class_flash_attn_func = flash_attn_func
                layer.attention.use_ring_attn = False
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_position_ids=encoder_position_ids,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                _, _, _, seqlens = _get_unpad_packing_data(attention_mask, return_seq_len=True)
                batch_cu_seq_lens_q = [
                    [0] + list(torch.cumsum(torch.tensor(x, dtype=torch.int32), dim=0).tolist())
                    for x in seqlens
                ]
                no_ring_hidden_states = outputs[0]
                no_ring_hidden_states = extract_local(no_ring_hidden_states, batch_cu_seq_lens_q, rank, world_size)
                no_ring_logits = self.output(no_ring_hidden_states)
                no_ring_logits = no_ring_logits.float()
                no_ring_all_hidden_states = outputs.hidden_states
                no_ring_all_hidden_states = [extract_local(v, batch_cu_seq_lens_q, rank, world_size) for v in no_ring_all_hidden_states]
            
                no_ring_loss = None
                if labels is not None:
                    # Shift so that tokens < n predict n
                    shift_logits = no_ring_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, self.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    no_ring_loss = loss_fct(shift_logits, shift_labels)
                print("no_ring_loss", no_ring_loss)
                
            
            self.config.attn_implementation = 'ring_flash_attn'
            for layer in self.model.layers:
                layer.attention.class_flash_attn_varlen_func = zigzag_ring_flash_attn_varlen_func
                layer.attention.class_flash_attn_func = zigzag_ring_flash_attn_func
                layer.attention.use_ring_attn = True
                layer.cross_attention.class_flash_attn_varlen_func = ring_flash_attn_varlen_func
                layer.cross_attention.class_flash_attn_func = ring_flash_attn_func
                layer.cross_attention.use_ring_attn = True
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_position_ids=encoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.output(hidden_states)
        logits = logits.float()
        
        if debug_ring_attention:
            all_hidden_states = outputs.hidden_states
            
            # compute difference between logits and hidden_states
            for no_ring_v, v in [(no_ring_logits, logits), (no_ring_hidden_states, hidden_states)]:
                assert no_ring_v.shape == v.shape, f"no_ring_v.shape: {no_ring_v.shape}, v.shape: {v.shape}"
                assert no_ring_v.dtype == v.dtype, f"no_ring_v.dtype: {no_ring_v.dtype}, v.dtype: {v.dtype}"
            diff = (no_ring_logits - logits).abs().mean()
            print(f"diff between logits: {diff}")
            diff = (no_ring_hidden_states - hidden_states).abs().mean()
            print(f"diff between hidden_states: {diff}")
            
            
            print("ring", [v.shape for v in all_hidden_states])
            print("no ring", [v.shape for v in no_ring_all_hidden_states])
            for i, (no_ring_v, v) in enumerate(zip(no_ring_all_hidden_states, all_hidden_states)):
                assert no_ring_v.shape == v.shape, f"no_ring_v.shape: {no_ring_v.shape}, v.shape: {v.shape}"
                assert no_ring_v.dtype == v.dtype, f"no_ring_v.dtype: {no_ring_v.dtype}, v.dtype: {v.dtype}"
                diff = (no_ring_v - v).abs().mean()
                print(f"diff between all_hidden_states[{i}]: {diff}")
                
            if torch.isnan(diff).any():
                print('ring hidden_states', hidden_states)
                print('torch.isnan(no_ring_hidden_states).any()', torch.isnan(no_ring_hidden_states).any())
                print('torch.isnan(hidden_states).any()', torch.isnan(hidden_states).any())
                # save attention mask, encoder_attention_mask, position_ids, encoder_position_ids
                torch.save(attention_mask, 'attention_mask.pt')
                torch.save(encoder_attention_mask, 'encoder_attention_mask.pt')
                torch.save(position_ids, 'position_ids.pt')
                torch.save(encoder_position_ids, 'encoder_position_ids.pt')
                raise ValueError('Hidden states contains nan')
        
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
            # check is loss is nan, then set it to zero
            if torch.isnan(loss).any():
                loss_fct = CrossEntropyLoss(reduction='sum')
                loss = loss_fct(shift_logits, shift_labels)
        # print("loss", loss)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        output['logits'] = output['logits'].to(device)
        return output

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2] # get the layer 0's k value 's sequence length
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            # input_ids = input_ids[:, remove_prefix_length:]
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get('position_ids', None)
        encoder_hidden_states = kwargs.get('encoder_hidden_states', None)
        encoder_attention_mask = kwargs.get('encoder_attention_mask', None)
        encoder_position_ids = kwargs.get('encoder_position_ids', None)
        
        if encoder_hidden_states is not None and encoder_position_ids is None:
            if encoder_attention_mask is not None:
                encoder_position_ids = encoder_attention_mask.long().cumsum(-1) - 1
                encoder_position_ids.masked_fill_(encoder_attention_mask == 0, 1)
            else:
                # create position_ids on the fly for batch generation
                encoder_position_ids = torch.arange(
                    encoder_hidden_states.shape[1], dtype=torch.long, device=encoder_hidden_states.device
                )
                encoder_position_ids = encoder_position_ids.unsqueeze(0)
        
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            if self.config.enable_shared_cross_attention and encoder_hidden_states is not None:
                position_ids += encoder_position_ids.max(-1).values.unsqueeze(0) # to debug
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}
        
        model_inputs.update(
            {
                'position_ids': position_ids,
                'past_key_values': past_key_values,
                'use_cache': kwargs.get('use_cache'),
                'attention_mask': attention_mask,
                'encoder_hidden_states': encoder_hidden_states,
                'encoder_attention_mask': encoder_attention_mask,
                'encoder_position_ids': encoder_position_ids,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = [], meta_instruction=''):
        if tokenizer.add_bos_token:
            prompt = ''
        else:
            prompt = tokenizer.bos_token
        if meta_instruction:
            prompt += f"""<|im_start|>system\n{meta_instruction}<|im_end|>\n"""
        for record in history:
            prompt += f"""<|im_start|>user\n{record[0]}<|im_end|>\n<|im_start|>assistant\n{record[1]}<|im_end|>\n"""
        prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
        return tokenizer([prompt], return_tensors='pt')

    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        query: str,
        history: List[Tuple[str, str]] = [],
        streamer: Optional[BaseStreamer] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.8,
        meta_instruction: str = 'You are an AI assistant whose name is InternLM (书生·浦语).\n'
        '- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
        '- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.',
        **kwargs,
    ):
        inputs = self.build_inputs(tokenizer, query, history, meta_instruction)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(['<|im_end|>'])[0]]
        outputs = self.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        outputs = outputs[0].cpu().tolist()[len(inputs['input_ids'][0]) :]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split('<|im_end|>')[0]
        history = history + [(query, response)]
        return response, history

    @torch.no_grad()
    def stream_chat(
        self,
        tokenizer,
        query: str,
        history: List[Tuple[str, str]] = [],
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.8,
        **kwargs,
    ):
        """
        Return a generator in format: (response, history)
        Eg.
        ('你好，有什么可以帮助您的吗', [('你好', '你好，有什么可以帮助您的吗')])
        ('你好，有什么可以帮助您的吗？', [('你好', '你好，有什么可以帮助您的吗？')])
        """
        if BaseStreamer is None:
            raise ModuleNotFoundError(
                'The version of `transformers` is too low. Please make sure '
                'that you have installed `transformers>=4.28.0`.'
            )

        response_queue = queue.Queue(maxsize=20)

        class ChatStreamer(BaseStreamer):
            def __init__(self, tokenizer) -> None:
                super().__init__()
                self.tokenizer = tokenizer
                self.queue = response_queue
                self.query = query
                self.history = history
                self.response = ''
                self.cache = []
                self.received_inputs = False
                self.queue.put((self.response, history + [(self.query, self.response)]))

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError('ChatStreamer only supports batch size 1')
                elif len(value.shape) > 1:
                    value = value[0]

                if not self.received_inputs:
                    # The first received value is input_ids, ignore here
                    self.received_inputs = True
                    return

                self.cache.extend(value.tolist())
                token = self.tokenizer.decode(self.cache, skip_special_tokens=True)
                if token.strip() != '<|im_end|>':
                    self.response = self.response + token
                    history = self.history + [(self.query, self.response)]
                    self.queue.put((self.response, history))
                    self.cache = []
                else:
                    self.end()

            def end(self):
                self.queue.put(None)

        def stream_producer():
            return self.chat(
                tokenizer=tokenizer,
                query=query,
                streamer=ChatStreamer(tokenizer=tokenizer),
                history=history,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

        def consumer():
            producer = threading.Thread(target=stream_producer)
            producer.start()
            while True:
                res = response_queue.get()
                if res is None:
                    return
                yield res

        return consumer()


# Copied from transformers.model.llama.modeling_llama.LlamaForSequenceClassification with Llama->InternLM2
@add_start_docstrings(
    """
    The InternLM2 Model transformer with a sequence classification head on top (linear layer).

    [`InternLM2ForSequenceClassification`] uses the last token in order to do the classification,
    as other causal models (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    InternLM2_START_DOCSTRING,
)
class InternLM2ForSequenceClassification(InternLM2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = InternLM2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_position_ids=encoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError('Cannot handle batch sizes > 1 if no padding token is defined.')
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'

            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
