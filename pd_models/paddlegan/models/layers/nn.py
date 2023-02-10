# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.nn as nn
import math

class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho
            w = w.clip(self.clip_min, self.clip_max)
            module.rho.set_value(w)

        # used for photo2cartoon training
        if hasattr(module, 'w_gamma'):
            w = module.w_gamma
            w = w.clip(self.clip_min, self.clip_max)
            module.w_gamma.set_value(w)

        if hasattr(module, 'w_beta'):
            w = module.w_beta
            w = w.clip(self.clip_min, self.clip_max)
            module.w_beta.set_value(w)
