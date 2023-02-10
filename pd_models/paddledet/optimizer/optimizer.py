# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.nn as nn

import paddle.optimizer as optimizer
from core.workspace import register


__all__ = ['OptimizerBuilder']

from utils.logger import setup_logger
logger = setup_logger(__name__)

@register
class OptimizerBuilder():
    """
    Build optimizer handles
    Args:
        regularizer (object): an `Regularizer` instance
        optimizer (object): an `Optimizer` instance
    """
    __category__ = 'optim'

    def __init__(self,
                 clip_grad_by_norm=None,
                 regularizer={'type': 'L2',
                              'factor': .0001},
                 optimizer={'type': 'Momentum',
                            'momentum': .9}):
        self.clip_grad_by_norm = clip_grad_by_norm
        self.regularizer = regularizer
        self.optimizer = optimizer

    def __call__(self, learning_rate, model=None):
        if self.clip_grad_by_norm is not None:
            grad_clip = nn.ClipGradByGlobalNorm(
                clip_norm=self.clip_grad_by_norm)
        else:
            grad_clip = None

        optim_args = self.optimizer.copy()
        optim_type = optim_args['type']
        del optim_args['type']

        op = getattr(optimizer, optim_type)

        if 'param_groups' in optim_args:
            assert isinstance(optim_args['param_groups'], list), ''

            param_groups = optim_args.pop('param_groups')

            params, visited = [], []
            for group in param_groups:
                assert isinstance(group,
                                  dict) and 'params' in group and isinstance(
                                      group['params'], list), ''
                _params = {
                    n: p
                    for n, p in model.named_parameters()
                    if any([k in n
                            for k in group['params']]) and p.trainable is True
                }
                _group = group.copy()
                _group.update({'params': list(_params.values())})

                params.append(_group)
                visited.extend(list(_params.keys()))

            ext_params = [
                p for n, p in model.named_parameters()
                if n not in visited and p.trainable is True
            ]

            if len(ext_params) < len(model.parameters()):
                params.append({'params': ext_params})

            elif len(ext_params) > len(model.parameters()):
                raise RuntimeError

        else:
            _params = model.parameters()
            params = [param for param in _params if param.trainable is True]

        return op(learning_rate=learning_rate,
                  parameters=params,
                  grad_clip=grad_clip,
                  **optim_args)