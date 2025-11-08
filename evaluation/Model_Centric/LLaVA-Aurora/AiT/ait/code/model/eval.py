import os.path as osp

import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import Hook, LoggerHook, get_dist_info
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.core import encode_mask_results
from mmdet.apis.test import collect_results_gpu
from mmcv.image import tensor2imgs


metrics = {
    'det': 'bbox',
    'insseg': ['bbox', 'segm'],
    'depth': None,
}


def multi_gpu_test(model, data_loaders, runner=None, logger=None, out_dir=None, show_score_thr=0.3):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.

    Returns:
        list: The prediction results.
    """
    model.eval()
    task_result = {}
    rank, world_size = get_dist_info()
    for task, data_loader in data_loaders.items():
        if rank == 0:
            print(f'\neval task {task}')
        results = []
        dataset = data_loader.dataset
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))
        time.sleep(2)  # This line can prevent deadlock problem in some cases.
        loader_indices = data_loader.batch_sampler
        for batch_indices, data in zip(loader_indices, data_loader):
            with torch.no_grad():
                result = model(return_loss=False,
                               rescale=True, task=task, **data)
                # result, binary_mask = model(return_loss=False, rescale=True, task=task, **data)
                # encode mask results
                if task == 'insseg':
                    if out_dir:  # show insseg
                        PALETTE = getattr(dataset, 'PALETTE', None)
                        CLASSES = getattr(dataset, 'CLASSES', None)
                        batch_size = len(result)
                        if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                            img_tensor = data['img'][0]
                        else:
                            img_tensor = data['img'][0].data[0]
                        img_metas = data['img_metas'][0].data[0]
                        imgs = tensor2imgs(
                            img_tensor, **img_metas[0]['img_norm_cfg'])
                        assert len(imgs) == len(img_metas)

                        for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                            h, w, _ = img_meta['img_shape']
                            img_show = img[:h, :w, :]

                            ori_h, ori_w = img_meta['ori_shape'][:-1]
                            img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                            out_file = osp.join(
                                out_dir, img_meta['ori_filename'])

                            model.module.show_insseg_result(
                                img_show,
                                result[i],
                                bbox_color=PALETTE,
                                text_color=PALETTE,
                                mask_color=PALETTE,
                                show=False,
                                out_file=out_file,
                                score_thr=show_score_thr,
                                class_names=CLASSES)
                    result = [(bbox_results, encode_mask_results(mask_results))
                              for bbox_results, mask_results in result]
            results.extend(result)

            if rank == 0:
                batch_size = len(result)
                for _ in range(batch_size * world_size):
                    prog_bar.update()

        # collect results from all ranks
        results = collect_results_gpu(results, len(dataset))
        if rank == 0:
            eval_res = data_loader.dataset.evaluate(
                results, logger=logger, metric=metrics[task])
            print(eval_res)
            if runner is not None:
                for name, val in eval_res.items():
                    runner.log_buffer.output[task + '_' + name] = val
    return


class DistEvalMultitaskHook(Hook):

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 broadcast_bn_buffer=True,
                 gpu_collect=True,
                 **eval_kwargs):
        # no super init
        if interval <= 0:
            raise ValueError(f'interval must be a positive number, '
                             f'but got {interval}')

        assert isinstance(by_epoch, bool), '``by_epoch`` should be a boolean'

        if start is not None and start < 0:
            raise ValueError(f'The evaluation start epoch {start} is smaller '
                             f'than 0')
        self.dataloader = dataloader
        self.interval = interval
        self.start = start
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs
        self.initial_flag = True

        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.gpu_collect = gpu_collect

    def before_train_iter(self, runner):
        """Evaluate the model only at the start of training by iteration."""
        if self.by_epoch or not self.initial_flag:
            return
        if self.start is not None and runner.iter >= self.start:
            self.after_train_iter(runner)
        self.initial_flag = False

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        if not (self.by_epoch and self.initial_flag):
            return
        if self.start is not None and runner.epoch >= self.start:
            self.after_train_epoch(runner)
        self.initial_flag = False

    def after_train_iter(self, runner):
        """Called after every training iter to evaluate the results."""
        if not self.by_epoch and self._should_evaluate(runner):
            # Because the priority of EvalHook is higher than LoggerHook, the
            # training log and the evaluating log are mixed. Therefore,
            # we need to dump the training log and clear it before evaluating
            # log is generated. In addition, this problem will only appear in
            # `IterBasedRunner` whose `self.by_epoch` is False, because
            # `EpochBasedRunner` whose `self.by_epoch` is True calls
            # `_do_evaluate` in `after_train_epoch` stage, and at this stage
            # the training log has been printed, so it will not cause any
            # problem. more details at
            # https://github.com/open-mmlab/mmsegmentation/issues/694
            for hook in runner._hooks:
                if isinstance(hook, LoggerHook):
                    hook.after_train_iter(runner)
            runner.log_buffer.clear()

            self._do_evaluate(runner)

    def after_train_epoch(self, runner):
        """Called after every training epoch to evaluate the results."""
        if self.by_epoch and self._should_evaluate(runner):
            self._do_evaluate(runner)

    def _should_evaluate(self, runner):
        """Judge whether to perform evaluation.

        Here is the rule to judge whether to perform evaluation:
        1. It will not perform evaluation during the epoch/iteration interval,
           which is determined by ``self.interval``.
        2. It will not perform evaluation if the start time is larger than
           current time.
        3. It will not perform evaluation when current time is larger than
           the start time but during epoch/iteration interval.

        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        if self.by_epoch:
            current = runner.epoch
            check_time = self.every_n_epochs
        else:
            current = runner.iter
            check_time = self.every_n_iters

        if self.start is None:
            if not check_time(runner, self.interval):
                # No evaluation during the interval.
                return False
        elif (current + 1) < self.start:
            # No evaluation if start is larger than the current time.
            return False
        else:
            # Evaluation only at epochs/iters 3, 5, 7...
            # if start==3 and interval==2
            if (current + 1 - self.start) % self.interval:
                return False
        return True

    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        for task_name, dataloader in self.dataloader.items():
            eval_res = dataloader.dataset.evaluate(
                results, logger=runner.logger, metric=self.metrics[task_name], **self.eval_kwargs)

            for name, val in eval_res.items():
                runner.log_buffer.output[task_name + '_' + name] = val
        runner.log_buffer.ready = True

        return None

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        multi_gpu_test(
            runner.model,
            self.dataloader,
            runner=runner,
            logger=runner.logger)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            runner.log_buffer.ready = True
