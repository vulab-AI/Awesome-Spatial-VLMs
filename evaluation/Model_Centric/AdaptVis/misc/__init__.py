import torch.nn as nn
import torch
import random
import numpy as np
import re
import os
import pickle
import collections
import PIL


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
# taken from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py and modified
def _default_collate(batch):
    r"""
        Function that takes in a batch of data and puts the elements within the batch
        into a tensor with an additional outer dimension - batch size. The exact output type can be
        a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
        Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
        This is used as the default function for collation when
        `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.
        Here is the general input type (based on the type of the element within the batch) to output type mapping:
            * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
            * NumPy Arrays -> :class:`torch.Tensor`
            * `float` -> :class:`torch.Tensor`
            * `int` -> :class:`torch.Tensor`
            * `str` -> `str` (unchanged)
            * `bytes` -> `bytes` (unchanged)
            * `Mapping[K, V_i]` -> `Mapping[K, _default_collate([V_1, V_2, ...])]`
            * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[_default_collate([V1_1, V1_2, ...]),
              _default_collate([V2_1, V2_2, ...]), ...]`
            * `Sequence[V1_i, V2_i, ...]` -> `Sequence[_default_collate([V1_1, V1_2, ...]),
              _default_collate([V2_1, V2_2, ...]), ...]`
        Args:
            batch: a single batch to be collated
        Examples:
            >>> # Example with a batch of `int`s:
            >>> _default_collate([0, 1, 2, 3])
            tensor([0, 1, 2, 3])
            >>> # Example with a batch of `str`s:
            >>> _default_collate(['a', 'b', 'c'])
            ['a', 'b', 'c']
            >>> # Example with `Map` inside the batch:
            >>> _default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
            {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
            >>> # Example with `NamedTuple` inside the batch:
            >>> # xdoctest: +SKIP
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> _default_collate([Point(0, 0), Point(1, 1)])
            Point(x=tensor([0, 1]), y=tensor([0, 1]))
            >>> # Example with `Tuple` inside the batch:
            >>> _default_collate([(0, 1), (2, 3)])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Example with `List` inside the batch:
            >>> _default_collate([[0, 1], [2, 3]])
            [tensor([0, 2]), tensor([1, 3])]
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return _default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    
    # For curious reader, this is what we added to the original code:
    elif (isinstance(elem, PIL.Image.Image) or isinstance(elem, PIL.JpegImagePlugin.JpegImageFile)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: _default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: _default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(_default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [_default_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([_default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [_default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def save_scores(scores, args):
    """
    Utility function to save retrieval scores.
    """
    si2t_file = os.path.join(args.output_dir, "scores", f"{args.dataset}--{args.text_perturb_fn}--{args.image_perturb_fn}--{args.model_name}--{args.seed}--si2t.pkl")
    st2i_file = os.path.join(args.output_dir, "scores", f"{args.dataset}--{args.text_perturb_fn}--{args.image_perturb_fn}--{args.model_name}--{args.seed}--st2i.pkl")
    
    if isinstance(scores, tuple):
        scores_i2t = scores[0]
        scores_t2i = scores[1].T # Make it N_ims x N_text
    
    else:
        scores_t2i = scores
        scores_i2t = scores
        
    os.makedirs(os.path.dirname(si2t_file), exist_ok=True)
    pickle.dump(scores_i2t, open(si2t_file, "wb"))
    pickle.dump(scores_t2i, open(st2i_file, "wb"))
