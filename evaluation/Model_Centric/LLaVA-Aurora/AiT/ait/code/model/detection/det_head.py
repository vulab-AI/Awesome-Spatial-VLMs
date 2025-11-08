# reproduce of https://arxiv.org/abs/2109.10852
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, BaseModule, auto_fp16

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.builder import HEADS

import torch.distributed as dist
import random
from timm.models.vision_transformer import trunc_normal_


@HEADS.register_module()
class DetHead(BaseModule):
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
                 seq_aug=None,
                 class_label_corruption='rand_n_fake_cls',
                 **kwargs):
        super().__init__(init_cfg)
        self.task_id = task_id
        self.loss_weight = loss_weight
        self.class_label_corruption = class_label_corruption
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        self.num_query = num_query
        self.num_bins = num_bins
        self.coord_norm = coord_norm
        self.norm_val = norm_val
        self.num_classes = num_classes
        self.fp16_enabled = False
        self.with_mask = with_mask

        self.transformer = transformer
        self.seq_aug = seq_aug
        self.num_vocal = self.transformer.num_vocal
        self.class_offset = num_bins + 1
        self.special_offset = self.class_offset + num_classes
        self.noise_label = self.special_offset + 1

        self.loss_seq = nn.CrossEntropyLoss(reduction='sum')

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
        max_len = max([t[0].size(0) for t in targets])
        for b_i, target in enumerate(targets):
            box, label, img_size = target
            h, w = img_size[0], img_size[1]
            scale_factor = torch.tensor([w, h, w, h], device=device)

            label_token = label.unsqueeze(1) + self.num_bins + 1
            if self.coord_norm == 'abs':
                norm_box = box / self.norm_val
            else:
                norm_box = box / scale_factor
            box_tokens = (
                norm_box * self.num_bins).round().long().clamp(min=0, max=self.num_bins)
            input_tokens = torch.cat([box_tokens, label_token], dim=1)
            if self.seq_aug is None:
                input_tokens = input_tokens.flatten()
                nan = torch.full((max_len * 5 - len(input_tokens),),
                                 0, dtype=torch.int64, device=device)
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
            bad_bbox_shift = box_tiled[torch.randperm(len(box_tiled))[
                :bad_bbox_size]]
            bad_bbox_shift = shift_bbox(bad_bbox_shift, scale_factor)
            bad_bbox_random = torch.cat(
                (random_bbox(bad_bbox_size, scale_factor), bad_bbox_shift), dim=0)
            bad_bbox = bad_bbox_random[torch.randperm(
                len(bad_bbox_random))[:bad_bbox_size]]
            bad_bbox_label = torch.full(
                (bad_bbox_size, ), random_class, device=device)

            # Create dup bbox.
            dup_bbox = box_tiled[torch.randperm(
                len(box_tiled), device=device)[:dup_bbox_size]]
            dup_bbox = jitter_bbox(dup_bbox, max_range=0.1)
            dup_bbox_label = torch.full(
                (dup_bbox_size, ), random_class, device=device)

            shuffle_idx = torch.randperm(num_noise)
            noise_box = torch.cat((bad_bbox, dup_bbox), dim=0)[shuffle_idx].max(
                box.new([0., 0., 0., 0.])).min(scale_factor)
            noise_box_label = torch.cat(
                (bad_bbox_label, dup_bbox_label), dim=0)[shuffle_idx]
            if self.coord_norm == 'abs':
                random_norm_box = noise_box / self.norm_val
            else:
                random_norm_box = noise_box / scale_factor
            noise_box_tokens = (
                random_norm_box * self.num_bins).round().long().clamp(min=0, max=self.num_bins)
            fake_tokens = torch.cat(
                [noise_box_tokens, noise_box_label.unsqueeze(1)], dim=1)

            input_seq = torch.cat([input_tokens, fake_tokens], dim=0).flatten()
            input_seq_list.append(input_seq)

        return torch.stack(input_seq_list, dim=0)

    def get_targets(self, input_seq, num_objects_list):
        N, L = input_seq.shape[0], input_seq.shape[1]//5
        for seq, num_objects in zip(input_seq, num_objects_list):
            seq.view(L, 5)[num_objects:, :4] = -100
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
            x, input_seq, masks, self.task_id, pred_len=self.num_query*5)

        return (outs_dec,)

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores_list,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None,
             input_seq=None):
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = all_cls_scores_list[-1]
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        target_seq = self.get_targets(
            input_seq, [len(gt_label) for gt_label in gt_labels_list])

        num_pos = (target_seq > -1).sum()
        num_pos = torch.as_tensor(
            [num_pos], dtype=torch.float, device=target_seq.device)
        if self.sync_cls_avg_factor and dist.is_available() and dist.is_initialized():
            torch.distributed.all_reduce(num_pos)
        if dist.is_available() and dist.is_initialized():
            num_pos = torch.clamp(
                num_pos / dist.get_world_size(), min=1).item()

        # only for log purpose
        B, L, C = all_cls_scores.shape
        all_cls_scores_objs = all_cls_scores.view(B, -1, 5, C)
        target_seq_objs = target_seq.view(B, -1, 5)

        all_cls_scores_box = all_cls_scores_objs[:,
                                                 :, :4, :].reshape(-1, self.num_vocal)
        target_seq_box = target_seq_objs[:, :, :4].flatten()
        num_pos_box = (target_seq_box > -1).sum()
        bbox_ce = self.loss_seq(
            all_cls_scores_box, target_seq_box) / num_pos_box.clamp(min=1)

        all_cls_scores_cls = all_cls_scores_objs[:,
                                                 :, 4, :].reshape(-1, self.num_vocal)
        target_seq_cls = target_seq_objs[:, :, 4].flatten()
        num_pos_cls = (target_seq_cls > -1).sum()
        cls_ce = self.loss_seq(
            all_cls_scores_cls, target_seq_cls) / num_pos_cls.clamp(min=1)

        all_cls_scores = all_cls_scores.reshape(-1, self.num_vocal)
        target_seq = target_seq.flatten()

        loss_seq = self.loss_seq(all_cls_scores, target_seq) / num_pos

        loss_dict = {'loss': loss_seq * self.loss_weight}
        loss_dict.update({'cls_ce': cls_ce, 'bbox_ce': bbox_ce})
        return loss_dict

    @force_fp32(apply_to=('out_seq',))
    def loss2(
        self,
        out_seq,
        target_seq
    ):
        # only for log purpose
        B, L, C = out_seq.shape
        out_seq_objs = out_seq.view(B, -1, 5, C)
        target_seq_objs = target_seq.view(B, -1, 5)

        out_seq_box = out_seq_objs[:, :, :4, :].reshape(-1, self.num_vocal)
        target_seq_box = target_seq_objs[:, :, :4].flatten()
        num_pos_box = (target_seq_box > -1).sum()
        bbox_ce = self.loss_seq(
            out_seq_box, target_seq_box) / num_pos_box.clamp(min=1)

        out_seq_cls = out_seq_objs[:, :, 4, :].reshape(-1, self.num_vocal)
        target_seq_cls = target_seq_objs[:, :, 4].flatten()
        num_pos_cls = (target_seq_cls > -1).sum()
        cls_ce = self.loss_seq(
            out_seq_cls, target_seq_cls) / num_pos_cls.clamp(min=1)

        out_seq = out_seq.reshape(-1, self.num_vocal)
        target_seq = target_seq.flatten()

        loss_seq = self.loss_seq(out_seq, target_seq)

        loss_dict = {'loss': loss_seq * self.loss_weight}
        loss_dict.update({'cls_ce': cls_ce, 'bbox_ce': bbox_ce})
        return loss_dict

    def corrupt_label(self, input_seq):
        if self.class_label_corruption == 'none':
            return input_seq
        N, L = input_seq.shape[0], input_seq.shape[1]//5
        device = input_seq.device
        randlabel = torch.randint(
            self.num_classes, (N, L), device=device) + self.num_bins + 1
        noiselabel = torch.full((N, L), self.noise_label,
                                device=device)  # noise label
        if self.class_label_corruption == 'rand_n_fake_cls':
            corruptlabel = torch.where(torch.rand(
                (N, L), device=device) < 0.5, randlabel, noiselabel)
        elif self.class_label_corruption == 'rand_cls':
            corruptlabel = randlabel
        elif self.class_label_corruption == 'real_n_fake_cls':
            reallabel = input_seq.view(N, L, 5)[:, :, 4]
            corruptlabel = torch.where(torch.rand(
                (N, L), device=device) < 0.5, reallabel, noiselabel)
        else:
            raise NotImplementedError
        input_seq.view(N, L, 5)[:, :, 4] = corruptlabel
        return input_seq

    def get_seq(self,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=None,
                proposal_cfg=None,
                ):
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        # random permute box
        for idx in range(len(gt_labels)):
            rand_idx = torch.randperm(
                len(gt_labels[idx])).to(gt_labels[idx].device)
            gt_labels[idx] = gt_labels[idx][rand_idx]
            gt_bboxes[idx] = gt_bboxes[idx][rand_idx]
        input_seq = self.build_input_seq(list(zip(gt_bboxes, gt_labels, [
                                         img_meta['img_shape'][:2] for img_meta in img_metas])), max_objects=self.num_query)
        input_seq_corruptlabel = self.corrupt_label(input_seq.clone())[:, :-1]
        target_seq = self.get_targets(
            input_seq, [len(gt_label) for gt_label in gt_labels])  # set -100 ignore

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
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           img_shape,
                           scale_factor,
                           rescale=False):
        pred_tokens, pred_scores = cls_score
        seq_len = pred_tokens.shape[0]
        if seq_len < 5:
            device, dtype = pred_tokens.device, pred_tokens.dtype
            return torch.tensor([], device=device, dtype=dtype), torch.tensor([], device=device, dtype=dtype)
        num_objects = seq_len // 5
        pred_tokens = pred_tokens[:int(
            num_objects * 5)].reshape(num_objects, 5)
        pred_boxes = pred_tokens[:, :4]
        pred_class = pred_tokens[:, 4]
        if self.coord_norm == 'abs':
            boxes_per_image = pred_boxes * self.norm_val / self.num_bins
        else:
            boxes_per_image = pred_boxes / self.num_bins
            boxes_per_image[:, 0::2] = boxes_per_image[:, 0::2] * img_shape[1]
            boxes_per_image[:, 1::2] = boxes_per_image[:, 1::2] * img_shape[0]
        if rescale:
            boxes_per_image /= boxes_per_image.new_tensor(scale_factor)

        return torch.cat((boxes_per_image, pred_scores[:num_objects, None]), dim=-1), pred_class

    def simple_test(self, feats, img_metas, rescale=False):
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list
