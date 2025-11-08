import os
import argparse
import random
import math
import torch

import torch.distributed as dist

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from timm.scheduler.cosine_lr import CosineLRScheduler
from mmcv.utils import Config, DictAction
from datetime import datetime

from utils.train_api import save_model
from utils.logger import get_logger

from dataset.nyudepthv2 import NYUDepthV2
from utils.metrics import eval_depth, cropping_img
from models.VQVAE import VQVAE

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


def validation(vae, val_data_loader, logger):
    vae.eval()
    with torch.no_grad():
        loss_all = 0.0
        recon_loss_all = 0.0
        result_metrics = {}
        for metric in metric_name:
            result_metrics[metric] = 0.0

        for iter, (image, _) in enumerate(val_data_loader):
            image = image.cuda()
            loss, recon_loss, recons = vae(
                img=image,
            )
            loss_all += loss.mean().item()
            recon_loss_all += recon_loss.mean().item()

            if iter % 100 == 0:
                print("Validation processed {0} of {1}".format(
                    iter, len(val_data_loader)))
            pred_d = recons.squeeze()
            image_gt = image.squeeze()
            pred_crop, gt_crop = cropping_img(pred_d, image_gt)

            computed_result = eval_depth(
                torch.flatten(pred_crop), torch.flatten(gt_crop))

            for key in result_metrics.keys():
                result_metrics[key] += computed_result[key]

    loss_all = loss_all / len(val_data_loader)
    recon_loss_all = recon_loss_all / len(val_data_loader)
    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (iter + 1)

    val_loss_message = {'val_loss': loss_all,
                        'val_recons_loss': recon_loss_all,
                        'val_d1': result_metrics['d1'],
                        'val_d2': result_metrics['d2'],
                        'val_abs_rel': result_metrics['abs_rel'],
                        'val_sq_rel': result_metrics['sq_rel'],
                        'val_rmse': result_metrics['rmse']
                        }

    logger.info(val_loss_message)
    torch.cuda.empty_cache()
    vae.train()


def train(vae, train_data_loader, val_data_loader, opt, sched, cfg, logger, output_dir):
    opt_params = cfg.train_setting.opt_params
    model_params = cfg.model
    EPOCHS = opt_params.epochs
    train_objective = model_params.train_objective

    total_step = 0
    for epoch in range(EPOCHS):
        train_data_loader.sampler.set_epoch(epoch)
        for iter, (image, mask_image) in enumerate(train_data_loader):
            if cfg.train_setting.data.mask:
                mask_image = mask_image.cuda()
                image = image.cuda()
                loss, recon_loss, _ = vae(
                    img=mask_image,
                    label_img=image,
                )
            else:
                image = image.cuda()
                loss, recon_loss, _ = vae(
                    img=image,
                )

            opt.zero_grad()
            loss.backward()
            opt.step()

            if dist.get_rank() == 0 and iter % 100 == 0:
                lr = opt.param_groups[0]['lr']
                losses = {
                    'loss': loss.mean().item(),
                    'recon_loss': recon_loss.mean().item(),
                }
                log_messages = {
                    'epoch': epoch,
                    'iter': "{0}/{1}".format(iter, len(train_data_loader)),
                    'lr': lr,
                }
                log_messages.update(losses)
                logger.info(log_messages)

            if iter % opt_params.schedule_step == 0:
                if isinstance(sched, torch.optim.lr_scheduler.ExponentialLR):
                    sched.step()
                else:
                    sched.step_update(total_step)
            torch.cuda.synchronize()
            total_step += 1

        if dist.get_rank() == 0:
            validation(vae, val_data_loader, logger)
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

    # training dataset & dataloader
    train_dataset_kwargs = cfg.train_setting.data
    train_dataset = NYUDepthV2(**train_dataset_kwargs)
    train_data_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    assert len(train_dataset) > 0, 'folder does not contain any images'
    if dist.get_rank() == 0:
        print(f'{len(train_dataset)} images found for training')

    opt_params = cfg.train_setting.opt_params
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_data_sampler,
        batch_size=opt_params.batch_size // num_tasks,
        num_workers=1,
        drop_last=True,
    )

    # validation dataset & dataloader
    val_dataset_kwargs = cfg.test_setting.data
    val_dataset = NYUDepthV2(**val_dataset_kwargs)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=None,
        batch_size=1,
        num_workers=1,
    )

    # optimizer
    opt = Adam(vae.parameters(), lr=opt_params.learning_rate)
    if opt_params.schedule_type == 'exp':
        sched = ExponentialLR(optimizer=opt, gamma=opt_params.lr_decay_rate)
    elif opt_params.schedule_type == 'cosine':
        sched = CosineLRScheduler(
            opt,
            t_initial=math.ceil(opt_params.epochs *
                                len(train_dataset) / opt_params.batch_size),
            lr_min=1e-6,
            warmup_lr_init=opt_params.warmup_lr_init,
            warmup_t=opt_params.warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

    if len(args.eval) == 0:
        train(vae, train_data_loader, val_data_loader,
              opt, sched, cfg, logger, output_dir)
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
            validation(vae, val_data_loader, logger)


if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
   
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                        'in xxx=yyy format will be merged into config file. If the value to '
                        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                        'Note that the quotation marks are necessary and that no white space '
                        'is allowed.')
    parser.add_argument('--eval', type=str, default='')
    args = parser.parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://', rank=args.local_rank)
    torch.distributed.barrier()

    main(cfg, args.local_rank)
