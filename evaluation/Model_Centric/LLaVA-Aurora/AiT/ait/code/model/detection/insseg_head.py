import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, BaseModule, auto_fp16
from mmcv.ops.roi_align import roi_align

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.builder import HEADS

import torch.distributed as dist
import random
from timm.models.vision_transformer import trunc_normal_

from model import build_model

import numpy as np
import os
from math import sqrt

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


@HEADS.register_module()
class InsSegHead(BaseModule):
    _version = 2

    def __init__(self,
                 task_id,
                 num_classes,
                 loss_weight=1.0,
                 num_query=100,
                 num_bins=2000,
                 coord_norm='abs',
                 norm_val=1333,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 with_mask=True,
                 init_cfg=None,
                 vae_cfg=dict(
                     token_length=16,
                     mask_size=64,
                     embedding_dim=64,
                     num_embeddings=256,
                     pretrained='vqvae_patch16_256_64.pth',
                     freeze=True
                 ),
                 seq_aug=None,
                 box_std=0.2,
                 cls_drop=0.,
                 mask_before_label=False,
                 sup_neg_mask=False,
                 no_sup_mask=False,
                 cls_weight=1.0,
                 box_weight=1.0,
                 mask_weight=1.0,
                 class_label_corruption='rand_n_fake_cls',
                 decoder_loss_weight=0.,
                 dice_loss_weight=1.0,
                 max_obj_decoderloss=100,
                 **kwargs):
        super().__init__(init_cfg)
        self.cnt = 0
        self.decoder_loss_weight = decoder_loss_weight
        self.max_obj_decoderloss = max_obj_decoderloss
        self.dice_loss_weight = dice_loss_weight
        self.task_id = task_id
        self.loss_weight = loss_weight
        self.class_label_corruption = class_label_corruption
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.mask_weight = mask_weight
        self.mask_before_label = mask_before_label
        self.sup_neg_mask = sup_neg_mask
        self.no_sup_mask = no_sup_mask
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        self.num_query = num_query
        self.num_bins = num_bins
        self.coord_norm = coord_norm
        self.norm_val = norm_val
        self.num_classes = num_classes
        self.fp16_enabled = False

        self.with_mask = with_mask
        self.box_std = box_std
        self.cls_drop = cls_drop
        self.transformer = transformer
        self.mask_token_length = vae_cfg['token_length']
        self.transformer.mask_token_length = self.mask_token_length
        self.vqvae = build_model(vae_cfg)
        self.mask_size = (vae_cfg['mask_size'], vae_cfg['mask_size'])
        self.vae_cfg = vae_cfg
        self.seq_aug = seq_aug
        self.num_vocal = self.transformer.num_vocal
        self.class_offset = num_bins + 1
        self.special_offset = self.class_offset + num_classes
        self.noise_label = self.special_offset + 1
        self.mask_offset = self.special_offset + 2

        class_weight = torch.ones(self.num_vocal)
        class_weight[:self.num_bins+1] = box_weight
        class_weight[self.class_offset: self.class_offset +
                     num_classes] = cls_weight
        class_weight[self.noise_label] = cls_weight
        class_weight[self.mask_offset: self.mask_offset +
                     self.vae_cfg.num_embeddings] = mask_weight

        self.loss_seq = nn.CrossEntropyLoss(
            weight=class_weight, reduction='sum')

    def build_input_seq(self, targets, max_objects=100):
        device = targets[0][0].device

        def shift_bbox(bbox, scale_factor):
            n = bbox.shape[0]
            box_xy = torch.rand((n, 2), device=device) * scale_factor[:2]
            box_wh = (bbox[:, 2:] - bbox[:, :2]) / 2.
            box = torch.cat([box_xy - box_wh, box_xy + box_wh], dim=-1)
            return box

        def random_bbox(n, scale_factor):
            box_xy = torch.rand((n, 2), device=device)
            # trunc normal generate [-2,2]
            box_wh = torch.abs(trunc_normal_(
                torch.empty((n, 2), device=device)) / 4)
            box = torch.cat([box_xy - box_wh, box_xy + box_wh],
                            dim=-1) * scale_factor
            return box

        def jitter_bbox(bbox, max_range=0.1):
            n = bbox.shape[0]
            if n == 0:
                return bbox
            w = bbox[:, 2] - bbox[:, 0]
            h = bbox[:, 3] - bbox[:, 1]

            noise = torch.stack([w, h, w, h], dim=-1)
            noise_rate = trunc_normal_(torch.empty(
                (n, 4), device=device)) / 2 * max_range
            bbox = bbox + noise * noise_rate

            return bbox

        assert self.coord_norm == 'abs'
        input_seq_list = []
        bad_bbox_flag_list = []
        max_len = max([t[0].size(0) for t in targets])
        for b_i, target in enumerate(targets):
            box, mask_token, gt_masks, mask_idx, label, img_size = target
            mask_token_sft = mask_token + self.mask_offset
            h, w = img_size[0], img_size[1]
            scale_factor = torch.tensor([w, h, w, h], device=device)

            label_token = label.unsqueeze(1) + self.class_offset
            if self.coord_norm == 'abs':
                norm_box = box / self.norm_val
            else:
                norm_box = box / scale_factor
            box_tokens = (
                norm_box * self.num_bins).round().long().clamp(min=0, max=self.num_bins)
            if self.mask_before_label:
                input_tokens = torch.cat(
                    [box_tokens, mask_token_sft, label_token], dim=1)
            else:
                input_tokens = torch.cat(
                    [box_tokens, label_token, mask_token_sft], dim=1)
            if self.seq_aug is None:
                input_tokens = input_tokens.flatten()
                nan = torch.full((max_len * (5 + self.vae_cfg.token_length) -
                                 len(input_tokens),), 0, dtype=torch.int64, device=device)
                input_seq = torch.cat([input_tokens, nan], dim=0)
                input_seq_list.append(input_seq)
                continue

            num_objects = input_tokens.shape[0]
            num_noise = max_objects - num_objects

            # Create bad bbox.
            dup_bbox_size = random.randint(0, num_noise)
            dup_bbox_size = 0 if num_objects == 0 else dup_bbox_size
            bad_bbox_size = num_noise - dup_bbox_size

            random_class = self.noise_label  # noise label
            multiplier = 1 if num_objects == 0 else num_noise // num_objects + 1
            box_tiled = box.repeat((multiplier, 1))
            box_tiled_idx = torch.arange(
                len(box), device=device).repeat(multiplier)
            bad_bbox_shift = box_tiled[torch.randperm(len(box_tiled))[
                :bad_bbox_size]]
            bad_bbox_shift = shift_bbox(bad_bbox_shift, scale_factor)
            bad_bbox_random = torch.cat(
                (random_bbox(bad_bbox_size, scale_factor), bad_bbox_shift), dim=0)
            bad_bbox = bad_bbox_random[torch.randperm(
                len(bad_bbox_random))[:bad_bbox_size]]
            bad_bbox_label = torch.full(
                (bad_bbox_size, ), random_class, device=device)
            bad_mask = self.vqvae.encode(torch.zeros(
                (1,)+self.mask_size, device=device))
            bad_mask_token = bad_mask.expand(bad_bbox_size, -1)

            # Create dup bbox.
            permidx = torch.randperm(len(box_tiled), device=device)[
                :dup_bbox_size]
            dup_bbox = box_tiled[permidx]
            dup_bbox = jitter_bbox(dup_bbox, max_range=0.1)
            dup_bbox = dup_bbox.max(box.new([0., 0., 0., 0.])).min(
                scale_factor)  # add for pre crop/roi
            dup_bbox_label = torch.full(
                (dup_bbox_size, ), random_class, device=device)

            dup_mask = crop_and_resize_gpu(
                gt_masks, dup_bbox, self.mask_size, mask_idx[box_tiled_idx[permidx]], device=device)
            dup_mask_token = self.vqvae.encode(dup_mask)

            shuffle_idx = torch.randperm(num_noise)
            bad_bbox_flag = torch.cat((torch.ones(bad_bbox_size, device=device), torch.zeros(
                dup_bbox_size, device=device))).to(torch.bool)
            bad_bbox_flag_list.append(
                bad_bbox_flag[shuffle_idx].nonzero(as_tuple=True)[0])
            noise_box = torch.cat((bad_bbox, dup_bbox), dim=0)[shuffle_idx].max(
                box.new([0., 0., 0., 0.])).min(scale_factor)
            noise_box_label = torch.cat(
                (bad_bbox_label, dup_bbox_label), dim=0)[shuffle_idx]
            noise_mask_token = torch.cat(
                (bad_mask_token, dup_mask_token), dim=0)[shuffle_idx]

            if self.coord_norm == 'abs':
                random_norm_box = noise_box / self.norm_val
            else:
                random_norm_box = noise_box / scale_factor
            noise_box_tokens = (
                random_norm_box * self.num_bins).round().long().clamp(min=0, max=self.num_bins)
            if self.mask_before_label:
                fake_tokens = torch.cat(
                    [noise_box_tokens, noise_mask_token + self.mask_offset, noise_box_label.unsqueeze(1)], dim=1)
            else:
                fake_tokens = torch.cat([noise_box_tokens, noise_box_label.unsqueeze(
                    1), noise_mask_token + self.mask_offset], dim=1)

            input_seq = torch.cat([input_tokens, fake_tokens], dim=0).flatten()
            input_seq_list.append(input_seq)
        return torch.stack(input_seq_list, dim=0), bad_bbox_flag_list

    def get_targets(self, input_seq, num_objects_list, bad_bbox_flag_list):
        N, L = input_seq.shape[0], input_seq.shape[1]//(
            5 + self.vae_cfg.token_length)
        for seq, num_objects, bad_bbox_flag in zip(input_seq, num_objects_list, bad_bbox_flag_list):
            seq.view(L, 5 + self.vae_cfg.token_length)[num_objects:, :4] = -100
            if self.no_sup_mask:
                if self.mask_before_label:
                    seq.view(
                        L, 5 + self.vae_cfg.token_length)[:, 4:4+self.vae_cfg.token_length] = -100
                else:
                    seq.view(L, 5 + self.vae_cfg.token_length)[:, 5:] = -100
            elif self.sup_neg_mask:
                if self.mask_before_label:
                    seq.view(L, 5 + self.vae_cfg.token_length)[
                        num_objects+bad_bbox_flag, 4:4+self.vae_cfg.token_length] = -100
                else:
                    seq.view(
                        L, 5 + self.vae_cfg.token_length)[num_objects+bad_bbox_flag, 5:] = -100
            else:
                if self.mask_before_label:
                    seq.view(
                        L, 5 + self.vae_cfg.token_length)[num_objects:, 4:4+self.vae_cfg.token_length] = -100
                else:
                    seq.view(
                        L, 5 + self.vae_cfg.token_length)[num_objects:, 5:] = -100

        return input_seq

    def forward(self, feats, img_metas, input_seq=None):
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        input_seq_list = [input_seq for _ in range(num_levels)]
        return multi_apply(self.forward_single, feats, img_metas_list, input_seq_list)

    def forward_single(self, x, img_metas, input_seq):
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        if self.with_mask:
            masks = x.new_ones((batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w, _ = img_metas[img_id]['img_shape']
                masks[img_id, :img_h, :img_w] = 0
        else:
            masks = x.new_zeros((batch_size, input_img_h, input_img_w))

        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        # outs_dec: [bs, num_query, embed_dim]
        outs_dec = self.transformer(
            x, input_seq, masks, self.task_id, pred_len=self.num_query*(5+self.mask_token_length))

        return (outs_dec,)

    @force_fp32(apply_to=('out_seq'))
    def loss2(self,
              out_seq,
              target_seq
              ):
        # NOTE defaultly only the outputs from the last feature scale is used.
        # only for log purpose
        B, L, C = out_seq.shape
        out_seq_objs = out_seq.view(B, -1, 5+self.mask_token_length, C)
        target_seq_objs = target_seq.view(B, -1, 5+self.mask_token_length)

        out_seq_box = out_seq_objs[:, :, :4, :].reshape(-1, self.num_vocal)
        target_seq_box = target_seq_objs[:, :, :4].flatten()
        num_pos_box = (target_seq_box > -1).sum()
        bbox_ce = self.loss_seq(
            out_seq_box, target_seq_box) / num_pos_box.clamp(min=1)

        if self.mask_before_label:
            out_seq_cls = out_seq_objs[:, :, -1, :].reshape(-1, self.num_vocal)
            target_seq_cls = target_seq_objs[:, :, -1].flatten()
            num_pos_cls = (target_seq_cls > -1).sum()
            cls_ce = self.loss_seq(
                out_seq_cls, target_seq_cls) / num_pos_cls.clamp(min=1)

            out_seq_mask = out_seq_objs[:, :, 4:4 +
                                        self.vae_cfg.token_length, :].reshape(-1, self.num_vocal)
            target_seq_mask = target_seq_objs[:, :,
                                              4:4+self.vae_cfg.token_length].flatten()
            num_pos_mask = (target_seq_mask > -1).sum()
            mask_ce = self.loss_seq(
                out_seq_mask, target_seq_mask) / num_pos_mask.clamp(min=1)
        else:
            out_seq_cls = out_seq_objs[:, :, 4, :].reshape(-1, self.num_vocal)
            target_seq_cls = target_seq_objs[:, :, 4].flatten()
            num_pos_cls = (target_seq_cls > -1).sum()
            cls_ce = self.loss_seq(
                out_seq_cls, target_seq_cls) / num_pos_cls.clamp(min=1)

            out_seq_mask = out_seq_objs[:, :, 5:,
                                        :].reshape(-1, self.num_vocal)
            target_seq_mask = target_seq_objs[:, :, 5:].flatten()
            num_pos_mask = (target_seq_mask > -1).sum()
            mask_ce = self.loss_seq(
                out_seq_mask, target_seq_mask) / num_pos_mask.clamp(min=1)
            num_obj = num_pos_mask // self.vae_cfg.token_length
            num_obj = num_obj.clamp(max=self.max_obj_decoderloss)
            if self.decoder_loss_weight > 0. and num_obj > 0.:
                assert B == 1
                vae_reso = int(sqrt(self.vae_cfg.token_length))
                vae_mask = out_seq_mask[:, self.mask_offset: self.mask_offset+self.vae_cfg.num_embeddings].view(
                    -1, vae_reso, vae_reso, self.vae_cfg.num_embeddings).permute(0, 3, 1, 2)
                soft_mask = self.vqvae.decode_soft(vae_mask[:num_obj])
                decoder_aux_ce = self.dice_loss_weight * \
                    dice_loss(
                        soft_mask, self.mask[:num_obj]) + F.mse_loss(soft_mask, self.mask[:num_obj])
            else:
                decoder_aux_ce = torch.tensor(
                    0., device=f'cuda:{torch.cuda.current_device()}')

            out_seq = out_seq.reshape(-1, self.num_vocal)
            target_seq = target_seq.flatten()

            loss_seq = self.loss_seq(out_seq, target_seq)

            loss_dict = {'loss': (loss_seq + decoder_aux_ce * self.decoder_loss_weight *
                                  num_pos_mask * self.mask_weight) * self.loss_weight}

            loss_dict.update({'cls_ce': cls_ce, 'bbox_ce': bbox_ce,
                             'mask_ce': mask_ce, 'decoder_aux_ce': decoder_aux_ce})
            return loss_dict

    def corrupt_label(self, input_seq):
        if self.class_label_corruption == 'none':
            return input_seq
        N, L = input_seq.shape[0], input_seq.shape[1]//(
            5+self.mask_token_length)
        device = input_seq.device
        randlabel = torch.randint(
            self.num_classes, (N, L), device=device) + self.class_offset
        noiselabel = torch.full((N, L), self.noise_label,
                                device=device)  # noise label
        if self.class_label_corruption == 'rand_n_fake_cls':
            corruptlabel = torch.where(torch.rand(
                (N, L), device=device) < 0.5, randlabel, noiselabel)
        elif self.class_label_corruption == 'rand_cls':
            corruptlabel = randlabel
        else:
            raise NotImplementedError
        if self.mask_before_label:
            input_seq.view(
                N, L, 5+self.mask_token_length)[:, :, -1] = corruptlabel
        else:
            input_seq.view(
                N, L, 5+self.mask_token_length)[:, :, 4] = corruptlabel
        return input_seq

    def get_seq(self,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_masks=None,
                gt_bboxes_ignore=None,
                proposal_cfg=None,
                ):
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        # random permute box
        device = gt_labels[0].device
        gt_masktoken_list = []
        rand_idx_list = []
        for idx in range(len(gt_labels)):
            rand_idx = torch.randperm(len(gt_labels[idx]), device=device)
            gt_labels[idx] = gt_labels[idx][rand_idx]
            gt_bboxes[idx] = gt_bboxes[idx][rand_idx]
            # crop mask
            mask = crop_and_resize_gpu(
                gt_masks[idx], gt_bboxes[idx], self.mask_size, rand_idx, device=device)
            self.mask = mask
            mask_token = self.vqvae.encode(mask)

            gt_masktoken_list.append(mask_token)
            rand_idx_list.append(rand_idx)
        input_seq, bad_bbox_flag_list = self.build_input_seq(list(zip(gt_bboxes, gt_masktoken_list, gt_masks, rand_idx_list, gt_labels, [
                                                             img_meta['img_shape'][:2] for img_meta in img_metas])), max_objects=self.num_query)
        input_seq_corruptlabel = self.corrupt_label(input_seq.clone())[:, :-1]
        target_seq = self.get_targets(input_seq, [len(
            gt_label) for gt_label in gt_labels], bad_bbox_flag_list)  # set -100 ignore

        return input_seq_corruptlabel, target_seq

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores_list,
                   img_metas,
                   rescale=False):
        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        cls_scores = all_cls_scores_list[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            img_shape = img_metas[img_id]['img_shape']
            ori_shape = img_metas[img_id]['ori_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score,
                                                img_shape, ori_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           img_shape,
                           ori_shape,
                           scale_factor,
                           rescale=False):
        if len(cls_score) == 2:
            pred_token, pred_score = cls_score
            pred_mask_logits = None
        else:
            pred_token, pred_score, pred_mask_logits = cls_score
        # pred_token = pred_token.long()
        seq_len = pred_token.shape[0]
        obj_len = 5 + self.mask_token_length
        if seq_len < obj_len:
            device, dtype = pred_score.device, pred_score.dtype
            return torch.tensor([], device=device, dtype=dtype), torch.tensor([], device=device, dtype=torch.long), torch.empty((0, ori_shape[0], ori_shape[1]), device=device, dtype=torch.bool)
        num_objects = seq_len // obj_len
        pred_token = pred_token[:int(
            num_objects * obj_len)].reshape(num_objects, obj_len)
        if self.mask_before_label:
            pred_bbox_token = pred_token[:, :4]
            pred_class_token = pred_token[:, -1].long()
            pred_mask_token = pred_token[:, 4:4 +
                                         self.vae_cfg.token_length].long()
        else:
            pred_bbox_token = pred_token[:, :4]
            pred_class_token = pred_token[:, 4].long()
            pred_mask_token = pred_token[:, 5:].long()
        if self.coord_norm == 'abs':
            boxes_per_image = pred_bbox_token * self.norm_val / self.num_bins
        else:
            boxes_per_image = pred_bbox_token / self.num_bins
            boxes_per_image[:, 0::2] = boxes_per_image[:, 0::2] * img_shape[1]
            boxes_per_image[:, 1::2] = boxes_per_image[:, 1::2] * img_shape[0]
        if rescale:
            boxes_per_image /= boxes_per_image.new_tensor(scale_factor)

        if pred_mask_logits is None:  # hard inf
            seg_masks = self.decode_mask(
                pred_mask_token, boxes_per_image, ori_shape, soft_inf=False)
        else:  # soft inf
            seg_masks = self.decode_mask(
                pred_mask_logits, boxes_per_image, ori_shape, soft_inf=True)

        return torch.cat((boxes_per_image, pred_score[:num_objects].unsqueeze(-1)), dim=-1), pred_class_token, seg_masks

    def decode_mask(self, mask_token, bboxes, ori_shape, soft_inf=False):
        img_h, img_w = ori_shape[:2]
        device = mask_token.device

        # N, 1, H, W
        if soft_inf:
            _N, _L, _C = mask_token.shape
            mask_token = mask_token.transpose(1, 2).reshape(
                _N, _C, int(sqrt(_L)), int(sqrt(_L)))
            mask_pred = self.vqvae.decode_soft(
                mask_token).unsqueeze(1)  # mask_token is mask_logits
        else:
            mask_pred = self.vqvae.decode(mask_token).unsqueeze(1)

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            # the types of img_w and img_h are np.int32,
            # when the image resolution is large,
            # the calculation of num_chunks will overflow.
            # so we need to change the types of img_w and img_h to int.
            # See https://github.com/open-mmlab/mmdetection/pull/5191
            num_chunks = int(
                np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT /
                        GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool)

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            # must use >0 because of grid_sample padding 0.
            # version change to >= 0.5 and add denorm before
            masks_chunk = (masks_chunk >= 0.5).to(dtype=torch.bool)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        return im_mask

    def simple_test(self, feats, img_metas, rescale=False):
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list


def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # inputs = inputs.sigmoid()
    # assert inputs.max() <= 1 and inputs.min() >= 0
    inputs = inputs.clip(min=0., max=1.)
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()


def crop_and_resize_gpu(bitmapmasks,
                        bboxes,
                        out_shape,
                        inds,
                        device='cpu',
                        interpolation='bilinear',
                        binarize=True):
    """See :func:`BaseInstanceMasks.crop_and_resize`."""
    if len(bitmapmasks.masks) == 0:
        return torch.empty((0, *out_shape), dtype=bboxes.dtype, device=device)

    # convert bboxes to tensor
    if isinstance(bboxes, np.ndarray):
        bboxes = torch.from_numpy(bboxes).to(device=device)
    if isinstance(inds, np.ndarray):
        inds = torch.from_numpy(inds).to(device=device)

    num_bbox = bboxes.shape[0]
    fake_inds = torch.arange(
        num_bbox, device=device).to(dtype=bboxes.dtype)[:, None]
    rois = torch.cat([fake_inds, bboxes], dim=1)  # Nx5
    rois = rois.to(device=device)
    if num_bbox > 0:
        gt_masks_th = torch.from_numpy(bitmapmasks.masks).to(device).index_select(
            0, inds).to(dtype=rois.dtype)
        targets = roi_align(gt_masks_th[:, None, :, :], rois, out_shape,
                            1.0, 0, 'avg', True).squeeze(1)
        if binarize:
            resized_masks = (targets >= 0.5)
        else:
            resized_masks = targets
    else:
        resized_masks = torch.empty(
            (0, *out_shape), dtype=bboxes.dtype, device=device)
    return resized_masks.to(dtype=bboxes.dtype)


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks according to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
