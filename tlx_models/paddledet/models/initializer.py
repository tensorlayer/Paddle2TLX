"""
This code is based on https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
Ths copyright of pytorch/pytorch is a BSD-style license, as found in the LICENSE file.
"""
import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
__all__ = ['constant_']


def _no_grad_fill_(tensor, value=0.0):
    tensor.set_value(tensorlayerx.constant(shape=tensor, fill_value=value,
        dtype=tensor.dtype))
    return tensor


def constant_(tensor, value=0.0):
    """
    Modified tensor inspace using constant_
    Args:
        tensor (paddle.Tensor): paddle Tensor
        value (float|int): value to fill tensor.
    Return:
        tensor
    """
    return _no_grad_fill_(tensor, value)
