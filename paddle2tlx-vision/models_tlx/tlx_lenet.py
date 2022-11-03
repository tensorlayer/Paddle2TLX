#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
os.environ['TL_BACKEND'] = 'paddle'
import paddle
# import paddle.nn as nn
import tensorlayerx as tlx
import tensorlayerx.nn as nn

__all__ = []


# class LeNet(nn.Layer):
class LeNet(nn.Module):
    """LeNet model from
    `"LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.`_

    Args:
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 10.

    Examples:
        .. code-block:: python

            from paddle.vision.models import LeNet

            model = LeNet()
    """

    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # nn.Conv2D(1, 6, 3, stride=1, padding=1),
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1, data_format='channels_first'),
            nn.ReLU(),
            # nn.MaxPool2D(2, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, data_format='channels_first'),
            # nn.Conv2D(6, 16, 5, stride=1, padding=0),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, data_format='channels_first'),
            nn.ReLU(),
            # nn.MaxPool2D(2, 2)
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, data_format='channels_first')
        )

        if num_classes > 0:
            self.fc = nn.Sequential(
                nn.Linear(in_features=400, out_features=120),
                nn.Linear(in_features=120, out_features=84),
                nn.Linear(in_features=84, out_features=num_classes)
            )

    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            # x = paddle.flatten(x, 1)
            x = nn.Flatten()(x)
            x = self.fc(x)
        return x
