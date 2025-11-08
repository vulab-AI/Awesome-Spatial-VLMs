import os
import math
import argparse
import random
import numpy as np

# torch
import torch
from torch.optim import AdamW
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ExponentialLR
from timm.scheduler.cosine_lr import CosineLRScheduler

from datetime import datetime
# vision imports

from torch.utils.data import DataLoader

from mmcv.utils import Config, DictAction

# dalle classes and utils

from utils.train_api import save_model, build_data_set
from utils.eval_coco import mask_evaluation
from utils.logger import get_logger
from models.VQVAE import VQVAE


def train(vae, train_data_loader, opt, sched, cfg, logger, output_dir):
    opt_params = cfg.train_setting.opt_params
    model_params = cfg.model
    EPOCHS = opt_params.epochs

    total_step = 0
    vae.train()
    for epoch in range(EPOCHS):
        for iter, (image, _) in enumerate(train_data_loader):
            image = image.cuda()
            loss, recon_loss, _ = vae(
                img=image,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

            if dist.get_rank() == 0 and iter % 100 == 0:
                lr = opt.param_groups[0]['lr']
                losses = {'loss': loss.mean().item(),
                          'recon_loss': recon_loss.mean().item(),
                          }
                log_messages = {
                    'epoch': epoch,
                    'iter': "{0}/{1}".format(iter, len(train_data_loader)),
                    'lr': lr,
                }
                log_messages.update(losses)
                logger.info(log_messages)

            if isinstance(sched, torch.optim.lr_scheduler._LRScheduler):
                sched.step()
            else:
                sched.step_update(total_step)
            torch.cuda.synchronize()
            total_step += 1

        if dist.get_rank() == 0:
            save_model(f'{output_dir}/vae_{epoch}.pt', vae)

    if dist.get_rank() == 0:
        save_model(f'{output_dir}/vae-final.pt', vae)


def main(cfg, local_rank=0):
    vae = VQVAE(
        **cfg.model,
    ).cuda()
    vae = torch.nn.parallel.DistributedDataParallel(
        vae, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)

    output_dir = cfg.train_setting.output_dir
    logger = None
    if dist.get_rank() == 0:
        print("creating logger")
        os.makedirs(output_dir, exist_ok=True)
        logger = get_logger(
            path="{0}/{1}.log".format(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S")))
        logger.info("begin log")
        logger.info(cfg)

    num_tasks = dist.get_world_size()
    print("num_tasks: {0}".format(num_tasks))
    global_rank = dist.get_rank()

    train_dataset = build_data_set(cfg.train_setting.data)
    data_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    opt_params = cfg.train_setting.opt_params
    train_data_loader = DataLoader(
        train_dataset, opt_params.batch_size // num_tasks, sampler=data_sampler)

    assert len(train_dataset) > 0, 'folder does not contain any images'

    # optimizer
    opt = AdamW(vae.parameters(), lr=opt_params.learning_rate,
                weight_decay=opt_params.weight_decay)
    if opt_params.schedule_type == 'exp':
        sched = ExponentialLR(optimizer=opt, gamma=opt_params.lr_decay_rate)
    elif opt_params.schedule_type == 'cosine':
        sched = CosineLRScheduler(
            opt,
            t_initial=math.ceil(opt_params.epochs *
                                len(train_dataset) / opt_params.batch_size),
            lr_min=1e-6,
            warmup_lr_init=opt_params.warmup_ratio * opt_params.learning_rate,
            warmup_t=opt_params.warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    if len(args.eval) == 0:
        train(vae, train_data_loader, opt, sched, cfg, logger, output_dir)
    else:
        ckpt = torch.load(args.eval)['weights']
        if 'module' not in list(ckpt.keys())[0]:
            new_ckpt = {}
            for key in list(ckpt.keys()):
                new_ckpt[f'module.{key}'] = ckpt[key]
            vae.load_state_dict(
                new_ckpt,
            )
        else:
            vae.load_state_dict(
                ckpt,
            )
    if dist.get_rank() == 0:
        mask_evaluation(vae, **cfg.test_setting)


if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument("--local_rank", type=int, required=True,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                        'in xxx=yyy format will be merged into config file. If the value to '
                        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                        'Note that the quotation marks are necessary and that no white space '
                        'is allowed.')
    parser.add_argument('--eval', type=str, default='')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://', rank=args.local_rank)
    torch.distributed.barrier()

    main(cfg, args.local_rank)
