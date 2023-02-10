#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is based on https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
Ths copyright of pytorch/pytorch is a BSD-style license, as found in the LICENSE file.
"""

import paddle

__all__ = [
    'constant_',
]

def _no_grad_fill_(tensor, value=0.):
    # with paddle.no_grad():
    tensor.set_value(paddle.full_like(shape=tensor, fill_value=value, dtype=tensor.dtype))
    return tensor


def constant_(tensor, value=0.):
    """
    Modified tensor inspace using constant_
    Args:
        tensor (paddle.Tensor): paddle Tensor
        value (float|int): value to fill tensor.
    Return:
        tensor
    """
    return _no_grad_fill_(tensor, value)
