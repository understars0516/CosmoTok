from functools import wraps
from typing import Sequence

import numpy as np
import torch
from numba import jit


@jit(nopython=True)
def is_sorted(a):
    for i in range(len(a) - 1):  # noqa: SIM110
        if a[i] > a[i + 1]:
            return False
    return True


@jit(nopython=True)
def all_same(a: np.ndarray):
    size = len(a)
    if size <= 0:
        return True
    for i in range(1, size):  # noqa: SIM110
        if a[i] != a[0]:
            return False
    return True


def merge_dict_without_duplicates(dest: dict, src: dict):
    """
    Merge two dictionaries, but raise error on duplicate keys.
    """
    for k, v in src.items():
        if k in dest:
            raise KeyError(f"{k} is already in the dict")
        dest[k] = v


def get_ddp_info():
    world_size = 1
    rank = 0
    local_rank = 0

    # 获取 DDP 分布式信息
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        local_rank = torch.distributed.get_node_local_rank()

    return world_size, rank, local_rank


def get_ddp_worker_info():
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return None, None
    return worker_info.num_workers, worker_info.id


def maybe_slice(x: "Sequence"):
    if len(x) <= 2:
        return x

    diff = np.diff(np.asarray(x))
    step = diff[0].item()
    if all_same(diff):
        return slice(x[0], x[-1] + step, step)
    else:
        return x


def center_crop(image, crop_size: int, leading_indexes=None):
    *_, height, width = image.shape
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2

    if leading_indexes is None:
        leading_indexes = ()
    elif not isinstance(leading_indexes, tuple):
        leading_indexes = (leading_indexes,)

    indexes = *leading_indexes, ..., slice(start_y, (start_y + crop_size)), slice(start_x, (start_x + crop_size))
    return image[indexes]


def inference_only(cls_or_func):

    def inference_only_forward_wrapper(forward_func):
        @wraps(forward_func)
        def inference_only_forward(self, *args, **kwargs):
            with torch.inference_mode():
                return forward_func(self, *args, **kwargs)

        return inference_only_forward

    def new_init_wrapper(original_init):
        @wraps(original_init)
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        return init

    if issubclass(cls_or_func, torch.nn.Module):
        cls_or_func.__init__ = new_init_wrapper(cls_or_func.__init__)
        cls_or_func.forward = inference_only_forward_wrapper(cls_or_func.forward)
    elif callable(cls_or_func):
        cls_or_func = inference_only_forward_wrapper(cls_or_func)
    else:
        raise TypeError(f"{cls_or_func} is not a callable or a subclass of torch.nn.Module")

    return cls_or_func


def _demo():
    print(is_sorted([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    print(maybe_slice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))


if __name__ == "__main__":
    _demo()
