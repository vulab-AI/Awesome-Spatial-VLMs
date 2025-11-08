import math
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from mmcv.runner import get_dist_info
from mmdet.models.builder import MODELS
from mmcv.runner.hooks import HOOKS, LrUpdaterHook
import numpy as np
import random
from functools import partial
from torch.utils.data import Sampler
import itertools
from mmdet.core.utils import sync_random_seed
from mmdet.datasets.builder import PIPELINES
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from mmcv.parallel.data_container import DataContainer

from mmdet.datasets import build_dataset, replace_ImageToTensor
from torch.utils.data.dataset import ConcatDataset
from collections import defaultdict


@PIPELINES.register_module()
class AddKey:
    def __init__(self, kv):
        self.kv = kv

    def __call__(self, results):
        for k, v in self.kv.items():
            results[k] = v
        return results


def annealing_linear(start, end, factor):
    """Calculate annealing linear learning rate.
    Linear anneal from `start` to `end` as percentage goes from 0.0 to 1.0.
    Args:
        start (float): The starting learning rate of the linear annealing.
        end (float): The ending learing rate of the linear annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
    """
    return start + (end - start) * factor


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@HOOKS.register_module()
class LinearAnnealingLrUpdaterHook(LrUpdaterHook):

    def __init__(self, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super(LinearAnnealingLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr
        return annealing_linear(base_lr, target_lr, progress / max_progress)


def build_model(cfg):
    """Build backbone."""
    return MODELS.build(cfg)


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 2.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        # self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.num_selected_samples = self.num_samples
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.seed+self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(2)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     repeat_aug=False,
                     is_train=True,
                     seed=None,
                     **kwargs):
    rank, world_size = get_dist_info()

    if is_train:
        if repeat_aug:
            sampler = RASampler(
                dataset, world_size, rank, seed=seed, shuffle=True)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, seed=seed, shuffle=True)
    else:
        sampler = DistributedSampler(
            dataset, world_size, rank, seed=seed, shuffle=False)

    init_fn = partial(
        worker_init_fn, num_workers=workers_per_gpu, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=samples_per_gpu,
        sampler=sampler,
        num_workers=workers_per_gpu,
        drop_last=True if is_train else False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


class MultitaskInfiniteBatchSampler(Sampler):
    """Similar to `BatchSampler` warping a `DistributedSampler. It is designed
    iteration-based runners like `IterBasedRunner` and yields a mini-batch
    indices each time.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (object): The dataset.
        batch_size (int): When model is :obj:`DistributedDataParallel`,
            it is the number of training samples on each GPU,
            When model is :obj:`DataParallel`, it is
            `num_gpus * samples_per_gpu`.
            Default : 1.
        world_size (int, optional): Number of processes participating in
            distributed training. Default: None.
        rank (int, optional): Rank of current process. Default: None.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the dataset or not. Default: True.
    """  # noqa: W605

    def __init__(self,
                 dataset,
                 batch_size=1,
                 world_size=None,
                 rank=None,
                 seed=0,
                 shuffle=True,
                 datasets_sizes=[0],
                 total_batches=[1]):
        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.batch_size = batch_size
        self.sizes = datasets_sizes
        self.datasets_start = np.cumsum([0]+datasets_sizes)[:-1]
        self.total_batches = total_batches
        assert sum(total_batches) == batch_size * world_size
        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)
        self.shuffle = shuffle
        self.size = len(dataset)
        self.indices = self._indices_of_rank()

    def _infinite_indices(self, size):
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)

        while True:
            if self.shuffle:
                yield from torch.randperm(size, generator=g).tolist()

            else:
                yield from torch.arange(size).tolist()

    def _merge_indices(self, indices_list):
        while True:
            for indices, bs, ind_start in zip(indices_list, self.total_batches, self.datasets_start):
                for i in range(bs):
                    yield next(indices) + ind_start

    def _indices_of_rank(self):
        """Slice the infinite indices by rank."""
        indices_list = [self._infinite_indices(sz) for sz in self.sizes]
        indices = self._merge_indices(indices_list)
        yield from itertools.islice(indices, self.rank, None,
                                    self.world_size)

    def __iter__(self):
        # once batch size is reached, yield the indices
        batch_buffer = []
        for idx in self.indices:
            batch_buffer.append(idx)
            if len(batch_buffer) == self.batch_size:
                yield batch_buffer
                batch_buffer = []

    def __len__(self):
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch):
        """Not supported in `IterationBased` runner."""
        raise NotImplementedError


def collate(batch, samples_per_gpu=0):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """
    samples_per_gpu = samples_per_gpu if samples_per_gpu > 0 else len(batch)
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


def collate_fn_wrapper(batch, collate_fn):
    data = defaultdict(list)
    task_type = ['det', 'insseg', 'depth']
    for b in batch:
        for t in task_type:
            if b['task_type'] == t:
                b.pop('task_type')
                data[t].append(b)
                break
    res = dict()
    for k, v in data.items():
        if k in ['insseg', 'det']:
            res[k] = collate_fn(v)
        else:
            res[k] = default_collate(v)
    return res


def build_dataloader_tasks(cfg, is_train=True):
    dataloaders = {}
    datasets_list = []
    if is_train:
        total_batches = []
        for task_name, task_cfg in cfg.task.items():
            if task_name in ['det', 'insseg']:
                if is_train:
                    data_cfg = task_cfg.data.train
                    samples_total_gpu = data_cfg.pop('samples_total_gpu')
                    dataset = build_dataset(data_cfg)
            elif task_name == 'depth':
                if is_train:
                    data_cfg = task_cfg.data.train
                    samples_total_gpu = data_cfg.pop('samples_total_gpu')
                    dataset = build_dataset(data_cfg)
            datasets_list.append(dataset)
            total_batches.append(samples_total_gpu)

        datasets = ConcatDataset(datasets_list)
        datasets_len = [len(ds) for ds in datasets_list]
        print('length:', datasets_len)
        rank, world_size = get_dist_info()
        assert sum(total_batches) % world_size == 0
        batch_sampler = MultitaskInfiniteBatchSampler(datasets, batch_size=sum(
            total_batches) / world_size, seed=cfg.seed, shuffle=True, datasets_sizes=datasets_len, total_batches=total_batches)
        dataloaders = DataLoader(datasets, batch_sampler=batch_sampler, num_workers=8, collate_fn=partial(
            collate_fn_wrapper, collate_fn=collate))
    else:
        for task_name, task_cfg in cfg.task.items():
            if task_name in ['det', 'insseg']:
                collate_fn = collate
                data_cfg = task_cfg.data.val
                samples_per_gpu = data_cfg.pop('samples_per_gpu')
                workers_per_gpu = data_cfg.pop('workers_per_gpu')
                if samples_per_gpu > 1:
                    # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                    data_cfg.pipeline = replace_ImageToTensor(
                        data_cfg.pipeline)
                dataset = build_dataset(data_cfg, dict(test_mode=True))
            elif task_name == 'depth':
                collate_fn = None
                data_cfg = task_cfg.data.val
                samples_per_gpu = data_cfg.pop('samples_per_gpu')
                workers_per_gpu = data_cfg.pop('workers_per_gpu')
                dataset = build_dataset(data_cfg)

            dataloaders[task_name] = build_dataloader(dataset, samples_per_gpu, workers_per_gpu, repeat_aug=False,
                                                      is_train=is_train, seed=cfg.seed, collate_fn=collate_fn, persistent_workers=False)

    return dataloaders
