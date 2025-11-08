import torch
import torch.nn.functional as F
from torch import nn
from .utils import MODELS

from timm.models.vision_transformer import DropPath, Mlp, trunc_normal_
import torch.utils.checkpoint as checkpoint
import math
from mmcv.runner import force_fp32, BaseModule, auto_fp16


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def with_pos(self, x, pos):
        return x if pos is None else x + pos

    def forward(self, q, k, v, mask=None, pre_kv=None, qpos=None, kpos=None):
        B, N, C = q.shape
        q = self.wq(self.with_pos(q, qpos)).reshape(
            B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.wk(self.with_pos(k, kpos)).reshape(
            B, k.shape[1], self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.wv(v).reshape(
            B, v.shape[1], self.num_heads, C // self.num_heads).transpose(1, 2)

        if pre_kv is not None:
            k = torch.cat([pre_kv[0], k], dim=2)
            v = torch.cat([pre_kv[1], v], dim=2)
            pre_kv = torch.stack([k, v], dim=0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn.masked_fill_(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x if pre_kv is None else (x, pre_kv)


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, drop_path=0.1, drop_out=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=(drop_out, 0.))

    def forward(self, x, mask=None, pos=None):
        norm_x = self.norm1(x)
        x = x + self.drop_path(self.attn(norm_x, norm_x,
                               norm_x, mask, qpos=pos, kpos=pos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, drop_path=0.1, drop_out=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        self.crossattn = Attention(dim, num_heads=num_heads)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.dropout = nn.Dropout(drop_out)
        self.norm3 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=(drop_out, 0.))

    def forward(self, x, z, mask1=None, mask2=None, pos=None, zpos=None, pre_kv=None):
        norm_x = self.norm1(x)
        if pre_kv is None:
            y = self.attn(norm_x, norm_x, norm_x, mask1, qpos=pos, kpos=pos)
        else:
            pos = None if pos is None else pos[:, -1:, :]
            y, pre_kv = self.attn(norm_x, norm_x, norm_x,
                                  mask1, pre_kv=pre_kv, qpos=pos, kpos=pos)
        x = x + self.drop_path(y)
        norm_x = self.norm2(x)
        norm_z = self.norm2(z)
        x = x + self.drop_path(self.crossattn(norm_x,
                               norm_z, norm_z, mask2, qpos=pos, kpos=zpos))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x if pre_kv is None else (x, pre_kv)


class Sequential(nn.Module):
    def __init__(self, *blocks, use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.use_checkpoint = use_checkpoint

    def __len__(self):
        return len(self.blocks)

    def forward(self, x, pre_kv_list=None, *args, **kwargs):
        use_checkpoint = self.use_checkpoint and self.training
        if pre_kv_list is None:
            for blk in self.blocks:
                if use_checkpoint:
                    assert len(kwargs) == 0
                    x = checkpoint.checkpoint(blk, x, *args, **kwargs)
                else:
                    x = blk(x, *args, **kwargs)
            return x
        else:
            cur_kv_list = []  # only use in eval
            for blk, pre_kv in zip(self.blocks, pre_kv_list):
                x, cur_kv = blk(x, *args, pre_kv=pre_kv, **kwargs)
                cur_kv_list.append(cur_kv)
            return x, cur_kv_list


@MODELS.register_module()
class ARTransformer(nn.Module):
    def __init__(self, in_chans, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, drop_path=0.1, drop_out=0.1, drop_path_linear=False,
                 num_vocal=2094, dec_length=2100, pred_eos=False,
                 num_bins=2001, num_classes=80, num_embeddings=128, num_embeddings_depth=128, checkpoint_encoder=False, checkpoint_decoder=False,
                 top_p=0.4, delay_eos=0, qk_pos=False, enc_mask=False, dec_mask=False,
                 pos_enc='sine', n_rows=20, n_cols=20, mask_before_label=False, ntasks=4, with_dec_embed=True, soft_vae=False, soft_transformer=False,
                 parallel=False, head_args={}
                 ):
        super().__init__()
        self.fp16_enabled = False
        self.head_args = head_args
        self.parallel = parallel
        self.soft_vae = soft_vae
        self.soft_transformer = soft_transformer
        self.num_vocal = num_vocal
        self.pred_eos = pred_eos
        self.top_p = top_p
        self.delay_eos = delay_eos
        self.qk_pos = qk_pos
        self.enc_mask = enc_mask
        self.dec_mask = dec_mask
        self.num_classes = num_classes
        self.mask_before_label = mask_before_label

        self.num_bins = num_bins
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.num_embeddings_depth = num_embeddings_depth

        self.class_offset = num_bins + 1
        self.special_offset = self.class_offset + num_classes
        self.noise_label = self.special_offset + 1
        self.mask_offset = self.special_offset + 2
        self.depth_offset = self.mask_offset + num_embeddings
        self.enddepth_offset = self.depth_offset + num_embeddings_depth

        self.nhead = nhead
        self.d_model = d_model

        self.n_rows = n_rows
        self.n_cols = n_cols

        self.add_enc_embed(pos_enc, n_rows, n_cols, d_model)

        if self.parallel:
            self.mask_embed = nn.Parameter(torch.empty(1, 1, d_model))
        self.det_embed = nn.Parameter(torch.empty(ntasks, 1, d_model))
        self.voc_embed = nn.Parameter(torch.empty(
            self.num_vocal - 2, d_model)) if pred_eos else nn.Parameter(torch.empty(self.num_vocal, d_model))
        self.with_dec_embed = with_dec_embed
        if with_dec_embed:
            self.dec_embed = nn.Parameter(
                torch.empty(ntasks, dec_length, d_model))

        self.input_proj = nn.Linear(in_chans, d_model)
        dpr = iter(torch.linspace(0, drop_path, num_encoder_layers).tolist(
        )) if drop_path_linear else iter([drop_path]*num_encoder_layers)
        self.encoder = Sequential(*[
            EncoderBlock(d_model, nhead, dim_feedforward,
                         drop_path=next(dpr), drop_out=drop_out)
            for _ in range(num_encoder_layers)
        ], use_checkpoint=checkpoint_encoder)
        dpr = iter(torch.linspace(0, drop_path, num_decoder_layers).tolist(
        )) if drop_path_linear else iter([drop_path]*num_decoder_layers)
        self.decoder = Sequential(*[
            DecoderBlock(d_model, nhead, dim_feedforward,
                         drop_path=next(dpr), drop_out=drop_out)
            for i in range(num_decoder_layers)
        ], use_checkpoint=checkpoint_decoder)

        self.norm = nn.LayerNorm(d_model)

        # self.vocal_classifier = nn.Linear(d_model, num_vocal)
        self.outp_bias = nn.Parameter(torch.empty(num_vocal))

        self.dropout = nn.Dropout(drop_out)
        self.stem_ln = nn.LayerNorm(d_model)
        self.encout_ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_ln = nn.LayerNorm(d_model)
        self.proj_mlp = Mlp(
            in_features=d_model, hidden_features=dim_feedforward, drop=(drop_out, 0.))
        self.proj_mlp_ln = nn.LayerNorm(d_model)
        self.proj_mlp_droppath = DropPath(drop_path)

        self.init_weights()

    def vocal_classifier(self, x):
        return x @ self.voc_embed.transpose(0, 1) + self.outp_bias

    def init_weights(self):
        trunc_normal_(self.det_embed, std=0.02)
        trunc_normal_(self.voc_embed, std=0.02)
        if hasattr(self, 'mask_embed'):
            trunc_normal_(self.mask_embed, std=0.02)
        if self.with_dec_embed:
            trunc_normal_(self.dec_embed, std=0.02)
        trunc_normal_(self.outp_bias, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def add_enc_embed(self, pos_enc, n_rows, n_cols, d_model):
        if pos_enc == 'sine':
            d_model /= 2
            y_embed = torch.arange(n_rows, dtype=torch.float32)
            x_embed = torch.arange(n_cols, dtype=torch.float32)
            dim_t = torch.arange(
                d_model, dtype=torch.float32)
            dim_t = 10000. ** (2 * (dim_t // 2) / d_model)
            pos_x = x_embed[:, None] / dim_t
            pos_y = y_embed[:, None] / dim_t
            pos_x = torch.stack(
                (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()),
                dim=-1).view(1, n_cols, -1).expand(n_rows, n_cols, -1)
            pos_y = torch.stack(
                (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()),
                dim=-1).view(n_rows, 1, -1).expand(n_rows, n_cols, -1)
            pos = torch.cat((pos_y, pos_x), dim=-1).view(1, n_rows, n_cols, -1)
            self.register_buffer("enc_embed", pos)
        elif pos_enc == 'sine_norm':
            d_model /= 2
            y_embed = torch.arange(n_rows, dtype=torch.float32)
            x_embed = torch.arange(n_cols, dtype=torch.float32)
            y_embed = y_embed / (n_rows-1) * 2 * math.pi
            x_embed = x_embed / (n_cols-1) * 2 * math.pi
            dim_t = torch.arange(
                d_model, dtype=torch.float32)
            dim_t = 10000. ** (2 * (dim_t // 2) / d_model)
            pos_x = x_embed[:, None] / dim_t
            pos_y = y_embed[:, None] / dim_t
            pos_x = torch.stack(
                (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()),
                dim=-1).view(1, n_cols, -1).expand(n_rows, n_cols, -1)
            pos_y = torch.stack(
                (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()),
                dim=-1).view(n_rows, 1, -1).expand(n_rows, n_cols, -1)
            pos = torch.cat((pos_y, pos_x), dim=-1).view(1, n_rows, n_cols, -1)
            self.register_buffer("enc_embed", pos)
        elif pos_enc == 'learned':
            self.enc_embed = nn.Parameter(
                torch.empty(1, n_rows, n_cols, d_model))
            trunc_normal_(self.enc_embed, std=0.02)
        else:
            raise ValueError('Unknown pos encoding %s' % pos_enc)

    @force_fp32()
    def forward(self, src, input_seq, mask, task_id, pred_len=0):
        """
        Args:
            src: shape[B, C, H, W]
            input_seq: shape[B, 501, C] for training and shape[B, 1, C] for inference
            mask: shape[B, H, W]
            pred_len is used for test only
        """
        H, W = src.shape[-2:]
        src = self.input_proj(self.dropout(src.flatten(2).transpose(1, 2)))
        src = self.stem_ln(src)
        mask = mask.flatten(1)[:, None, None, :] if (
            self.enc_mask or self.dec_mask) else None
        enc_mask = mask if self.enc_mask else None

        B, N, C = src.shape
        enc_embed = self.enc_embed[:, :H, :W, :].flatten(1, 2)
        z = self.encoder(src, None, enc_mask, enc_embed) \
            if self.qk_pos else self.encoder(src + enc_embed, None, enc_mask)
        z = self.encout_ln(z)
        # Add (optional) positional embedding to encoded visual units.
        z = self.proj_ln(self.proj(z)) + enc_embed
        z = z + self.proj_mlp_droppath(self.proj_mlp(self.proj_mlp_ln(z)))

        dec_mask = mask if self.dec_mask else None
        if self.training:
            # if task_id != 3:
            B = input_seq.shape[0]
            if z.shape[0] != B:
                z = z.repeat_interleave(B // z.shape[0], dim=0)
            M = input_seq.shape[1] + 1
            if self.with_dec_embed:
                pos = self.dec_embed[task_id, :M, :]
            input_embed = self.voc_embed[input_seq]
            if self.parallel:
                assert task_id[0] == 2
                input_embed = self.mask_embed.expand_as(input_embed)

            input_embed = torch.cat([
                self.det_embed[task_id, :, :],
                input_embed,
            ], dim=1)
            self_attn_mask = torch.triu(torch.ones(
                M, M, device=z.device), diagonal=1).bool()

            x = self.decoder(
                input_embed, None, z, self_attn_mask, dec_mask
            ) if not self.with_dec_embed else self.decoder(
                input_embed, None, z, self_attn_mask, dec_mask, pos, enc_embed
            ) if self.qk_pos else self.decoder(
                input_embed + pos, None, z, self_attn_mask, dec_mask
            )

            x = self.norm(x)
            pred_seq_logits = self.vocal_classifier(x)
            return pred_seq_logits
        else:
            if task_id == 0 or task_id == 1:  # det or insseg
                mask_token_length = 0 if task_id == 0 else self.mask_token_length  # got by insseghead
                end = torch.zeros(B, device=z.device).bool()
                end_lens = torch.zeros(B, device=z.device).long()
                obj_len = 5 + mask_token_length

                input_embed = self.det_embed[[task_id]].expand(B, -1, -1)
                pre_kv_lsit = [
                    torch.empty(
                        (2, B, self.nhead, 0, self.d_model // self.nhead),
                        device=z.device, dtype=torch.float32
                    )
                    for _ in range(len(self.decoder))
                ]
                pred_tokens = []
                pred_scores = []
                pred_mask_logits = []
                pred_box_logits = []

                def is_label(i):
                    if self.mask_before_label:
                        return (i-1) % obj_len == obj_len-1
                    else:
                        return (i-1) % obj_len == 4
                for i in range(1, pred_len + 1):
                    if self.with_dec_embed:
                        pos = self.dec_embed[[task_id], :i, :]
                    x, pre_kv_lsit = self.decoder(
                        input_embed, pre_kv_lsit, z, None, dec_mask
                    ) if not self.with_dec_embed else self.decoder(
                        input_embed, pre_kv_lsit, z, None, dec_mask, pos, enc_embed
                    ) if self.qk_pos else self.decoder(
                        input_embed + pos[:, -1:,
                                          :], pre_kv_lsit, z, None, dec_mask
                    )
                    x = self.norm(x)
                    logits = self.vocal_classifier(x)[:, -1, :]

                    is_mask_flg = False
                    is_label_flg = False
                    is_box_flg = False
                    if is_label(i):  # label
                        is_label_flg = True
                        offset = self.class_offset
                        offset_end = self.class_offset + self.num_classes
                        current_logits = logits[:,
                                                self.class_offset: self.class_offset + self.num_classes]
                        if self.pred_eos:
                            current_scores = current_logits.softmax(dim=-1)
                        else:
                            current_scores = torch.cat([current_logits, logits[:, [self.noise_label]]], dim=1).softmax(
                                dim=-1)[:, :-1]  # add noise label
                    elif (i-1) % obj_len < 4:  # box
                        is_box_flg = True
                        offset = 0
                        offset_end = self.num_bins+1
                        current_logits = logits[:, :self.num_bins+1]
                    else:  # mask
                        is_mask_flg = True
                        offset = self.mask_offset
                        offset_end = self.mask_offset+self.num_embeddings
                        current_logits = logits[:,
                                                self.mask_offset: self.mask_offset+self.num_embeddings]
                        if self.soft_vae or self.soft_transformer:
                            tmp_current_logits = current_logits.clone()
                            pred_mask_logits.append(tmp_current_logits)

                    top_p = self.top_p

                    if self.pred_eos and (i - 1) % obj_len == 0:
                        current_logits = torch.cat(
                            [current_logits, logits[:, [self.special_offset+1]] - self.delay_eos], dim=1)

                    # Sort logits in descending order to determine the nucleus.
                    sorted_logits, sorted_idx = torch.sort(
                        current_logits, descending=True)

                    # Get cumulative softmax probabilites. For every instance in batch, a
                    #  variable amount of tokens (N) will consitute the nucleus.
                    # shape: (batch_size, num_classes)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Determine indices of tokens at the tail of distribution. These will be
                    # removed from the nucleus.
                    sorted_idx_to_remove = cumulative_probs > top_p

                    # Shift the indices to the right to keep the first token outside nucleus.
                    sorted_idx_to_remove[...,
                                         1:] = sorted_idx_to_remove[..., :-1].clone()
                    sorted_idx_to_remove[..., 0] = 0

                    # Set logits to large negative value to avoid sampling them. Iterate over
                    # the batch of examples.
                    for t in range(current_logits.size()[0]):
                        idx_to_remove = sorted_idx[t][sorted_idx_to_remove[t]]
                        current_logits[t][idx_to_remove] = -float('Inf')

                    # Sample from the filtered distribution.
                    # shape: (batch_size, num_classes)
                    current_probs = F.softmax(current_logits, dim=-1)

                    # shape: (batch_size, )
                    current_predictions = torch.multinomial(current_probs, 1)
                    current_predictions = current_predictions.view(B)

                    if self.pred_eos and (i - 1) % obj_len == 0:
                        stop_state = current_predictions.eq(
                            current_logits.shape[-1] - 1)
                        end_lens += i * (~end * stop_state)
                        end = (stop_state + end).bool()
                        if end.all():
                            break
                    pred_token = current_predictions[:, None]
                    if is_label(i):  # label
                        pred_scores.append(torch.gather(
                            current_scores, 1, pred_token))

                    if self.soft_transformer and is_mask_flg:
                        input_embed = tmp_current_logits.softmax(
                            dim=1) @ self.voc_embed[offset: offset_end]
                        input_embed = input_embed.unsqueeze(1)
                    else:
                        input_embed = self.voc_embed[(pred_token + offset)]
                    pred_tokens.append(pred_token)

                if not self.pred_eos:
                    end_lens.fill_(pred_len)
                else:
                    end_lens[end_lens == 0] = self.dec_embed.size(1) - 1
                pred_tokens = torch.cat(pred_tokens, dim=1) \
                    if len(pred_tokens) > 0 else torch.tensor([], device=src.device).view(B, 0)
                pred_scores = torch.cat(pred_scores, dim=1) \
                    if len(pred_scores) > 0 else torch.tensor([], device=src.device).view(B, 0)

                if self.soft_vae:
                    pred_mask_logits = torch.stack(pred_mask_logits, dim=1)
                    Bs, L, C = pred_mask_logits.shape
                    pred_mask_logits = pred_mask_logits.view(
                        Bs, -1, mask_token_length, C)
                    pred_tokens = [(psl[:end_idx], score, mask_logit) for end_idx, psl, score, mask_logit in zip(
                        end_lens, pred_tokens, pred_scores, pred_mask_logits)]
                else:
                    pred_tokens = [(psl[:end_idx], score) for end_idx, psl, score in zip(
                        end_lens, pred_tokens, pred_scores)]
                return pred_tokens

            elif task_id == 2:  # depth
                input_embed = self.det_embed[[task_id]].expand(B, -1, -1)
                pre_kv_lsit = [
                    torch.empty(
                        (2, B, self.nhead, 0, self.d_model // self.nhead),
                        device=z.device, dtype=torch.float32
                    )
                    for _ in range(len(self.decoder))
                ]
                pred_tokens = []
                pred_logits = []
                for i in range(1, pred_len + 1):
                    if self.with_dec_embed:
                        pos = self.dec_embed[[task_id], :i, :]
                    x, pre_kv_lsit = self.decoder(
                        input_embed, pre_kv_lsit, z, None, dec_mask
                    ) if not self.with_dec_embed else self.decoder(
                        input_embed, pre_kv_lsit, z, None, dec_mask, pos, self.enc_embed
                    ) if self.qk_pos else self.decoder(
                        input_embed + pos[:, -1:,
                                          :], pre_kv_lsit, z, None, dec_mask
                    )
                    x = self.norm(x)
                    logits = self.vocal_classifier(x)[:, -1, :]

                    # varify prob
                    # varlogits, varidx = torch.sort(logits.softmax(dim=1), dim=1, descending=True)
                    # varifycumsum(varlogits, varidx, self.depth_offset-2, self.num_vocal-2)

                    current_logits = logits[:,
                                            self.depth_offset: self.enddepth_offset]
                    tmp_logits = current_logits.clone()

                    # Sort logits in descending order to determine the nucleus.
                    sorted_logits, sorted_idx = torch.sort(
                        current_logits, descending=True)

                    # Get cumulative softmax probabilites. For every instance in batch, a
                    #  variable amount of tokens (N) will consitute the nucleus.
                    # shape: (batch_size, num_classes)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Determine indices of tokens at the tail of distribution. These will be
                    # removed from the nucleus.
                    sorted_idx_to_remove = cumulative_probs > self.top_p

                    # Shift the indices to the right to keep the first token outside nucleus.
                    sorted_idx_to_remove[...,
                                         1:] = sorted_idx_to_remove[..., :-1].clone()
                    sorted_idx_to_remove[..., 0] = 0

                    # Set logits to large negative value to avoid sampling them. Iterate over
                    # the batch of examples.
                    for t in range(current_logits.size()[0]):
                        idx_to_remove = sorted_idx[t][sorted_idx_to_remove[t]]
                        current_logits[t][idx_to_remove] = -float('Inf')

                    # Sample from the filtered distribution.
                    # shape: (batch_size, num_classes)
                    current_probs = F.softmax(current_logits, dim=-1)

                    # shape: (batch_size, )
                    current_predictions = torch.multinomial(current_probs, 1)
                    current_predictions = current_predictions.view(B)

                    pred_token = current_predictions[:, None]
                    pred_tokens.append(current_predictions)
                    pred_logits.append(tmp_logits)

                    if self.parallel:
                        # TODO: rewrite to actually parallel to accelerate code
                        input_embed = self.mask_embed.expand(B, -1, -1)
                    elif self.soft_transformer:
                        input_embed = tmp_logits.softmax(
                            dim=1) @ self.voc_embed[self.depth_offset: self.enddepth_offset]
                        input_embed = input_embed.unsqueeze(1)
                    else:
                        input_embed = self.voc_embed[pred_token +
                                                     self.depth_offset]

                pred_logits = torch.stack(pred_logits, dim=1)
                pred_tokens = torch.stack(pred_tokens, dim=1)
                return pred_logits, pred_tokens

            else:
                raise NotImplementedError
