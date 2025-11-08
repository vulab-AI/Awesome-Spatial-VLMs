import torch

from .utils import MODELS
from mmcv.runner import BaseModule, auto_fp16
from mmdet.core import bbox2result
from .utils import build_model
from collections import OrderedDict
import torch.distributed as dist
import matplotlib.pyplot as plt
import torch.nn.functional as F
from operator import itemgetter
import mmcv
import numpy as np
from mmdet.core.visualization import imshow_det_bboxes


def mask2result(masks, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    cls_segms = [[] for _ in range(num_classes)
                 ]  # BG is not included in num_classes
    N = masks.shape[0]
    for i in range(N):
        cls_segms[labels[i]].append(masks[i].detach().cpu().numpy())
    return cls_segms


@MODELS.register_module()
class AiT(BaseModule):

    def __init__(self,
                 backbone,
                 transformer,
                 task_heads={'det': {}, 'insseg': {}, 'depth': {}},
                 padto=0,
                 resizeto=0,
                 loss_exclude_offset=0.,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.loss_exclude_offset = loss_exclude_offset
        self.fp16_enabled = False
        self.padto = padto
        self.resizeto = resizeto
        self.task2id = {k: v['task_id'] for k, v in task_heads.items()}
        self.backbone = build_model(backbone)
        transformer['head_args'] = task_heads
        transformer = build_model(transformer)
        self.transformer = transformer
        tasks = list(task_heads.keys())
        # TODO: do not pass the transformer into submodule
        for k, v in task_heads.items():
            if k == 'det':
                det_head = v.copy()
                det_head['transformer'] = transformer
                self.det_head = build_model(det_head)
            elif k == 'insseg':
                insseg_head = v.copy()
                insseg_head['transformer'] = transformer
                self.insseg_head = build_model(insseg_head)
            elif k == 'depth':
                depth_head = v.copy()
                depth_head['transformer'] = transformer
                self.depth_head = build_model(depth_head)
            else:
                raise NotImplementedError(f'{k} is not implemented now')

        self.tasks = tasks

    def get_seq(self, gt, task):
        if task == 'det':
            seq = self.det_head.get_seq(**gt)
        elif task == 'insseg':
            seq = self.insseg_head.get_seq(**gt)
        elif task == 'depth':
            seq = self.depth_head.get_seq(**gt)
        else:
            raise NotImplementedError(f'{task} is not implemented now')
        return seq

    def cattomax(self, seqs):
        maxlen_input, maxlen_target = 0, 0
        for seq in seqs:
            input_seq, target_seq = seq
            maxlen_input = max(maxlen_input, input_seq.shape[-1])
            maxlen_target = max(maxlen_target, target_seq.shape[-1])
        pad_input_seqs, pad_target_seqs = [], []
        for input_seq, target_seq in seqs:
            input_seq = F.pad(input_seq, (0, maxlen_input -
                              input_seq.shape[-1]), value=0)
            target_seq = F.pad(
                target_seq, (0, maxlen_target-target_seq.shape[-1]), value=-100)
            pad_input_seqs.append(input_seq)
            pad_target_seqs.append(target_seq)

        return torch.cat(pad_input_seqs), torch.cat(pad_target_seqs)

    def forward_train(self,
                      img,
                      **kwargs):
        img = torch.cat([self.resize(i) for i in img], dim=0)
        x = self.backbone(img)
        gt = kwargs['gt']
        task = kwargs['task']
        seqs = []
        task_ids = []
        split_size = []
        for i in range(len(gt)):
            seq = self.get_seq(gt[i], task=task[i])
            seqs.append(seq)
            bs = seq[-1].shape[0]
            split_size.append(bs)
            task_ids.extend(bs*[self.task2id[task[i]]])

        target_before_padding = map(itemgetter(1), seqs)

        seqs = self.cattomax(seqs)
        input_seq, target_seq = seqs
        task_ids = input_seq.new_tensor(task_ids)
        src = x[0]
        out = self.transformer(src, input_seq, None, task_ids)
        task_spec_outs = torch.split(out, split_size)
        # task_spec_targets = torch.split(target_seq, split_size)
        losses = {
            'det_loss': torch.tensor(0., device=f'cuda:{torch.cuda.current_device()}'), 'det_cls_ce': torch.tensor(0., device=f'cuda:{torch.cuda.current_device()}'), 'det_bbox_ce': torch.tensor(0., device=f'cuda:{torch.cuda.current_device()}'),
            'insseg_loss': torch.tensor(0., device=f'cuda:{torch.cuda.current_device()}'), 'insseg_cls_ce': torch.tensor(0., device=f'cuda:{torch.cuda.current_device()}'), 'insseg_bbox_ce': torch.tensor(0., device=f'cuda:{torch.cuda.current_device()}'), 'insseg_mask_ce': torch.tensor(0., device=f'cuda:{torch.cuda.current_device()}'), 'insseg_decoder_aux_ce': torch.tensor(0., device=f'cuda:{torch.cuda.current_device()}'),
            'depth_loss': torch.tensor(0., device=f'cuda:{torch.cuda.current_device()}'), 'depth_ce': torch.tensor(0., device=f'cuda:{torch.cuda.current_device()}'), 'depth_decoder_aux_ce': torch.tensor(0., device=f'cuda:{torch.cuda.current_device()}'),
        }
        norm_factor = {t: torch.tensor(0., device=img.device)
                       for t in self.tasks}
        for t, out, target in zip(task, task_spec_outs, target_before_padding):
            loss_dict = self.loss(t, out[..., :target.shape[-1], :], target)
            num_pos = (target > -1).sum()  # fix bug: target_seq -> target
            norm_factor[t] += num_pos
            for k, v in loss_dict.items():
                losses[f'{t}_{k}'] = v  # add task prefix

        num_pos_tasks = torch.stack(list(norm_factor.values()))
        dist.all_reduce(num_pos_tasks)
        num_pos_tasks = torch.clamp(
            num_pos_tasks / dist.get_world_size(), min=1)
        for t, n in zip(self.tasks, num_pos_tasks):
            losses[f'{t}_loss'] = losses[f'{t}_loss'] / n

        return losses

    def loss(self, task, out, target):
        if task == 'det':
            losses = self.det_head.loss2(out, target)
        elif task == 'insseg':
            if self.loss_exclude_offset != 0.:
                losses = self.insseg_head.loss3(
                    out, target, l=self.loss_exclude_offset)
            else:
                losses = self.insseg_head.loss2(out, target)
        elif task == 'depth':
            if self.loss_exclude_offset != 0.:
                losses = self.depth_head.loss3(
                    out, target, l=self.loss_exclude_offset)
            else:
                losses = self.depth_head.loss2(out, target)
        return losses

    def forward_test(self, imgs, img_metas=None, task='det', **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        if task in ['det', 'insseg']:
            for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError(
                        f'{name} must be a list, but got {type(var)}')

            num_augs = len(imgs)
            if num_augs != len(img_metas):
                raise ValueError(f'num of augmentations ({len(imgs)}) '
                                 f'!= num of image meta ({len(img_metas)})')

            for img, img_meta in zip(imgs, img_metas):
                batch_size = len(img_meta)
                for img_id in range(batch_size):
                    img_meta[img_id]['batch_input_shape'] = tuple(
                        img.size()[-2:])

            if num_augs == 1:
                if 'proposals' in kwargs:
                    kwargs['proposals'] = kwargs['proposals'][0]
                if task == 'det':
                    return self.simple_test_box(imgs[0], img_metas[0], **kwargs)
                else:
                    return self.simple_test_insseg(imgs[0], img_metas[0], **kwargs)
            else:
                raise NotImplementedError
        elif task == 'depth':
            depth_gt = kwargs['depth']
            depth_gt = depth_gt.squeeze(1)
            if self.depth_head.shift_window_test:
                bs, _, h, w = imgs.shape
                interval_all = w - h
                interval = interval_all // (self.depth_head.shift_size-1)
                sliding_images = []
                sliding_masks = torch.zeros((bs, 1, h, w), device=imgs.device)
                for i in range(self.depth_head.shift_size):
                    sliding_images.append(
                        imgs[..., :, i*interval:i*interval+h])
                    sliding_masks[..., :, i*interval:i*interval+h] += 1
                imgs = torch.cat(sliding_images, dim=0)
            if self.depth_head.flip_test:
                imgs = torch.cat((imgs, torch.flip(imgs, [3])), dim=0)

            feat = self.backbone(self.resize(imgs))
            pred_d = self.depth_head.simple_test(feat, **kwargs).unsqueeze(1)

            if self.depth_head.flip_test:
                batch_s = pred_d.shape[0]//2
                pred_d = (pred_d[:batch_s] +
                          torch.flip(pred_d[batch_s:], [3]))/2.0
            assert self.depth_head.shift_window_test == True
            if self.depth_head.shift_window_test:
                pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
                for i in range(self.depth_head.shift_size):
                    pred_s[..., :, i*interval:i*interval +
                           h] += pred_d[i*bs:(i+1)*bs]
                pred_d = pred_s/sliding_masks

            return list(zip(pred_d.squeeze(1).clip(min=1e-3).detach().cpu(), depth_gt.cpu()))

        else:
            raise NotImplementedError

    def resize(self, img):
        _, _, H, W = img.shape
        if self.resizeto > 0:
            img = F.interpolate(img, self.resizeto, mode='bilinear')
        if self.padto > 0:
            img = F.pad(img, (0, self.padto-W, 0, self.padto-H))
        return img

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)

    def simple_test_box(self, img, img_metas, rescale=False):
        feat = self.backbone(img)
        results_list = self.det_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.det_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test_insseg(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.backbone(img)
        results_list = self.insseg_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.insseg_head.num_classes)
            for det_bboxes, det_labels, _ in results_list
        ]
        mask_results = [
            mask2result(det_masks, det_labels, self.insseg_head.num_classes)
            for _, det_labels, det_masks in results_list
        ]
        return list(zip(bbox_results, mask_results))

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        # log_vars = {f'{task}_{k}':v for k,v in log_vars.items()}

        if isinstance(data['img'], torch.Tensor):
            num_samples = data['img'].shape[0]
        else:
            num_samples = sum(map(lambda x: x.shape[0], data['img']))
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def show_insseg_result(self,
                           img,
                           result,
                           score_thr=0.3,
                           bbox_color=(72, 101, 241),
                           text_color=(72, 101, 241),
                           mask_color=None,
                           thickness=2,
                           font_size=13,
                           win_name='',
                           show=False,
                           wait_time=0,
                           class_names=None,
                           out_file=None):
        """Draw `result` over `img`.
        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=class_names,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img
