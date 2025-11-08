import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, wrap_fp16_model, load_checkpoint

from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.utils import collect_env, get_root_logger

from model import build_model, build_dataloader_tasks, DistEvalMultitaskHook

from mmdet.core import build_optimizer
from runner import IterBasedRunnerMultitask
from mmcv.runner import build_runner, Fp16OptimizerHook, OptimizerHook

from model.eval import multi_gpu_test

import cv2
cv2.setNumThreads(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--eval', help='checkpoint for eval only')
    parser.add_argument('--visualize', help='image to visualize')
    parser.add_argument('--show-dir', help='show dir')
    parser.add_argument('--show-score-thr', help='show dir',
                        default=0.3, type=float)
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-auto-resume',
        action='store_true')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        help='training tasks',
        default=['det', 'insseg', 'depth'])
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--test-dist', action='store_true')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    args.auto_resume = not args.no_auto_resume
    cfg.auto_resume = args.auto_resume
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    if args.test_dist:
        return
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    dump_name = osp.join(cfg.work_dir, osp.basename(args.config))
    attempt_count = 10
    file_exists = False
    while attempt_count > 0 and not file_exists:
        try:
            cfg.dump(dump_name)
            file_exists = True
        except:
            pass
        attempt_count -= 1
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    # do not sync seed when repeat aug.
    # set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # build model
    model = build_model(cfg.model)
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=False)

    tmp_task = {}
    for tname, t in cfg.task.items():
        if t['times'] >= 1:
            tmp_task[tname] = t
    cfg.task = tmp_task
    if args.eval is not None:  # eval only
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model.module)
        load_checkpoint(model.module, args.eval, map_location='cpu')
        dataloaders = build_dataloader_tasks(cfg, is_train=False)
        if args.visualize is not None:
            from mmdet.datasets.pipelines import Compose
            from mmcv.parallel import collate
            from mmcv.image import tensor2imgs

            model.eval()
            transforms = Compose(cfg.insseg_test_pipeline)
            data = transforms({
                'img_prefix': None,
                'img_info' : {'filename': args.visualize}
                })
            with torch.no_grad():
                data = collate([data])
                PALETTE = getattr(dataloaders['insseg'].dataset, 'PALETTE', None)
                CLASSES = getattr(dataloaders['insseg'].dataset, 'CLASSES', None)
                result = model(return_loss=False,
                               rescale=True, task='insseg', **data)
                model.module.show_insseg_result(
                    args.visualize,
                    result[0],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=False,
                    out_file='work_dirs/show.png',
                    score_thr=0.3,
                    class_names=CLASSES)
                return
        multi_gpu_test(model, dataloaders, logger=logger,
                       out_dir=args.show_dir, show_score_thr=args.show_score_thr)
        return

    # build dataset
    dataloaders = build_dataloader_tasks(cfg)

    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register hooks
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    # cumulative_iters = sum(dict(task_times).values())
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(
            **cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # register validation
    if not args.no_validate:
        val_dataloaders = build_dataloader_tasks(cfg, is_train=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = False
        eval_hook = DistEvalMultitaskHook
        runner.register_hook(
            eval_hook(val_dataloaders, **eval_cfg), priority='LOW')

    # run model
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.get("auto_resume", False) and osp.exists(osp.join(runner.work_dir, 'latest.pth')):
        runner.resume(osp.join(runner.work_dir, 'latest.pth'))
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(dataloaders)


if __name__ == '__main__':
    main()
