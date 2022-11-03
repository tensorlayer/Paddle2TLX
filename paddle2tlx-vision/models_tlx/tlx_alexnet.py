# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append(".")
os.environ['TL_BACKEND'] = 'paddle'
import math
import paddle
# import paddle.nn as nn
import tensorlayerx.nn as nn
import paddle.nn.functional as F

# from paddle.nn import Linear, Dropout, ReLU
from tensorlayerx.nn import Linear, Dropout, ReLU
# from paddle.nn import Conv2D, MaxPool2D
from tensorlayerx.nn import Conv2d, MaxPool2d
# from paddle.nn.initializer import Uniform
from tensorlayerx.nn.initializers import random_uniform
# from paddle.fluid.param_attr import ParamAttr
from paddle.utils.download import get_weights_path_from_url
from utils.load_model import restore_model

model_urls = {
    "alexnet": (
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/AlexNet_pretrained.pdparams",
        "7f0f9f737132e02732d75a1459d98a43",
    )
}

__all__ = []


# class ConvPoolLayer(nn.Layer):
class ConvPoolLayer(nn.Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 filter_size,
                 stride,
                 padding,
                 stdv,
                 groups=1,
                 act=None):
        super(ConvPoolLayer, self).__init__()

        self.relu = ReLU() if act == "relu" else None

        # TODO - paddle: NCHW, tlx: NHWC
        # self._conv = Conv2D(
        #     in_channels=input_channels,
        #     out_channels=output_channels,
        #     kernel_size=filter_size,
        #     stride=stride,
        #     padding=padding,
        #     groups=groups,
        #     weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
        #     bias_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))
        self._conv = Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            data_format='channels_first',
            W_init=random_uniform(-stdv, stdv),
            b_init=random_uniform(-stdv, stdv)
        )  # TODO - TypeError: 'ParamAttr' object is not callable
        # self._pool = MaxPool2D(kernel_size=3, stride=2, padding=0)
        self._pool = MaxPool2d(kernel_size=3, stride=2, padding=0, data_format='channels_first')

    def forward(self, inputs):
        x = self._conv(inputs)
        if self.relu is not None:
            x = self.relu(x)
        x = self._pool(x)
        return x


# class AlexNet(nn.Layer):
class AlexNet(nn.Module):
    """AlexNet model from
    `"ImageNet Classification with Deep Convolutional Neural Networks"
    <https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`_

    Args:
        num_classes (int): Output dim of last fc layer. Default: 1000.

    Examples:
        .. code-block:: python

            from paddle.vision.models import AlexNet

            alexnet = AlexNet()

    """

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        stdv = 1.0 / math.sqrt(3 * 11 * 11)
        self._conv1 = ConvPoolLayer(3, 64, 11, 4, 2, stdv, act="relu")
        stdv = 1.0 / math.sqrt(64 * 5 * 5)
        self._conv2 = ConvPoolLayer(64, 192, 5, 1, 2, stdv, act="relu")
        stdv = 1.0 / math.sqrt(192 * 3 * 3)
        # self._conv3 = Conv2D(
        #     192,
        #     384,
        #     3,
        #     stride=1,
        #     padding=1,
        #     weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
        #     bias_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))
        self._conv3 = Conv2d(
            in_channels=192,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1,
            data_format='channels_first',
            W_init=random_uniform(-stdv, stdv),
            b_init=random_uniform(-stdv, stdv)
        )
        stdv = 1.0 / math.sqrt(384 * 3 * 3)
        # self._conv4 = Conv2D(
        #     384,
        #     256,
        #     3,
        #     stride=1,
        #     padding=1,
        #     weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
        #     bias_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))
        self._conv4 = Conv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            data_format='channels_first',
            W_init=random_uniform(-stdv, stdv),
            b_init=random_uniform(-stdv, stdv)
        )
        stdv = 1.0 / math.sqrt(256 * 3 * 3)
        self._conv5 = ConvPoolLayer(256, 256, 3, 1, 1, stdv, act="relu")

        if self.num_classes > 0:
            stdv = 1.0 / math.sqrt(256 * 6 * 6)
            # self._drop1 = Dropout(p=0.5, mode="downscale_in_infer")
            self._drop1 = Dropout(p=0.5)
            # self._fc6 = Linear(
            #     in_features=256 * 6 * 6,
            #     out_features=4096,
            #     weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
            #     bias_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))
            self._fc6 = Linear(
                in_features=256 * 6 * 6,
                out_features=4096,
                W_init=random_uniform(-stdv, stdv),
                b_init=random_uniform(-stdv, stdv)
            )

            # self._drop2 = Dropout(p=0.5, mode="downscale_in_infer")
            self._drop2 = Dropout(p=0.5)
            # self._fc7 = Linear(
            #     in_features=4096,
            #     out_features=4096,
            #     weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
            #     bias_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))
            self._fc7 = Linear(
                in_features=4096,
                out_features=4096,
                W_init=random_uniform(-stdv, stdv),
                b_init=random_uniform(-stdv, stdv)
            )
            # self._fc8 = Linear(
            #     in_features=4096,
            #     out_features=num_classes,
            #     weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
            #     bias_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))
            self._fc8 = Linear(
                in_features=4096,
                out_features=num_classes,
                W_init=random_uniform(-stdv, stdv),
                b_init=random_uniform(-stdv, stdv)
            )

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)
        x = self._conv3(x)
        x = F.relu(x)
        x = self._conv4(x)
        x = F.relu(x)
        x = self._conv5(x)

        if self.num_classes > 0:
            # x = paddle.flatten(x, start_axis=1, stop_axis=-1)
            x = nn.Flatten()(x)
            # x = self._drop1(x)
            x = self._fc6(x)
            x = F.relu(x)
            # x = self._drop2(x)
            x = self._fc7(x)
            x = F.relu(x)
            x = self._fc8(x)

        return x


def _alexnet(arch, pretrained, **kwargs):
    model = AlexNet(**kwargs)

    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        # model.load_dict(param)
        # model.set_dict(param)
        restore_model(param, model)

    return model


def alexnet(pretrained=False, **kwargs):
    """AlexNet model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            from paddle.vision.models import alexnet

            # build model
            model = alexnet()

            # build model and load imagenet pretrained weight
            # model = alexnet(pretrained=True)
    """
    return _alexnet('alexnet', pretrained, **kwargs)