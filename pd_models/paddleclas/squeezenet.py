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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.nn import Conv2D#, Dropout
from paddle.nn import AdaptiveAvgPool2D#, MaxPool2D
from paddle.fluid.param_attr import ParamAttr
# from ops.tlx_extend import tlx_Dropout, tlx_MaxPool2d
from paddle2tlx.pd2tlx.utils import load_model_clas

__all__ = []

model_urls = {
    'squeezenet1_0':
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SqueezeNet1_0_pretrained.pdparams',
     '30b95af60a2178f03cf9b66cd77e1db1'),
    'squeezenet1_1':
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SqueezeNet1_1_pretrained.pdparams',
     'a11250d3a1f91d7131fd095ebbf09eee'),
}


class MakeFireConv(nn.Layer):

    def __init__(self, input_channels, output_channels, filter_size, padding=0):
        super(MakeFireConv, self).__init__()
        self._conv = Conv2D(input_channels,
                            output_channels,
                            filter_size,
                            padding=padding,
                            weight_attr=ParamAttr(),
                            bias_attr=ParamAttr())

    def forward(self, x):
        x = self._conv(x)
        x = F.relu(x)
        return x


class MakeFire(nn.Layer):

    def __init__(self, input_channels, squeeze_channels, expand1x1_channels,
                 expand3x3_channels):
        super(MakeFire, self).__init__()
        self._conv = MakeFireConv(input_channels, squeeze_channels, 1)
        self._conv_path1 = MakeFireConv(squeeze_channels, expand1x1_channels, 1)
        self._conv_path2 = MakeFireConv(squeeze_channels,
                                        expand3x3_channels,
                                        3,
                                        padding=1)

    def forward(self, inputs):
        x = self._conv(inputs)
        x1 = self._conv_path1(x)
        x2 = self._conv_path2(x)
        return paddle.concat([x1, x2], axis=1)


class SqueezeNet(nn.Layer):
    """SqueezeNet model from
    `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/pdf/1602.07360.pdf>`_

    Args:
        version (str): version of squeezenet, which can be "1.0" or "1.1".
        num_classes (int): output dim of last fc layer. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import SqueezeNet

            # build v1.0 model
            model = SqueezeNet(version='1.0')

            # build v1.1 model
            # model = SqueezeNet(version='1.1')

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)

    """

    def __init__(self, version, num_classes=1000, with_pool=True):
        super(SqueezeNet, self).__init__()
        self.version = version
        self.num_classes = num_classes
        self.with_pool = with_pool

        supported_versions = ['1.0', '1.1']
        assert version in supported_versions, \
            "supported versions are {} but input version is {}".format(
                supported_versions, version)

        if self.version == "1.0":
            self._conv = Conv2D(3,
                                96,
                                7,
                                stride=2,
                                weight_attr=ParamAttr(),
                                bias_attr=ParamAttr())
            self._pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=0)
            self._conv1 = MakeFire(96, 16, 64, 64)
            self._conv2 = MakeFire(128, 16, 64, 64)
            self._conv3 = MakeFire(128, 32, 128, 128)
            self._conv4 = MakeFire(256, 32, 128, 128)
            self._conv5 = MakeFire(256, 48, 192, 192)
            self._conv6 = MakeFire(384, 48, 192, 192)
            self._conv7 = MakeFire(384, 64, 256, 256)
            self._conv8 = MakeFire(512, 64, 256, 256)
        else:
            self._conv = Conv2D(3,
                                64,
                                3,
                                stride=2,
                                padding=1,
                                weight_attr=ParamAttr(),
                                bias_attr=ParamAttr())
            self._pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=0)
            self._conv1 = MakeFire(64, 16, 64, 64)
            self._conv2 = MakeFire(128, 16, 64, 64)
            self._conv3 = MakeFire(128, 32, 128, 128)
            self._conv4 = MakeFire(256, 32, 128, 128)
            self._conv5 = MakeFire(256, 48, 192, 192)
            self._conv6 = MakeFire(384, 48, 192, 192)
            self._conv7 = MakeFire(384, 64, 256, 256)
            self._conv8 = MakeFire(512, 64, 256, 256)

        self._drop = nn.Dropout(p=0.5, mode="downscale_in_infer")  # todo
        self._conv9 = Conv2D(512,
                             num_classes,
                             1,
                             weight_attr=ParamAttr(),
                             bias_attr=ParamAttr())
        self._avg_pool = AdaptiveAvgPool2D(1)

    def forward(self, inputs):
        x = self._conv(inputs)
        x = F.relu(x)
        x = self._pool(x)
        if self.version == "1.0":
            x = self._conv1(x)
            x = self._conv2(x)
            x = self._conv3(x)
            x = self._pool(x)
            x = self._conv4(x)
            x = self._conv5(x)
            x = self._conv6(x)
            x = self._conv7(x)
            x = self._pool(x)
            x = self._conv8(x)
        else:
            x = self._conv1(x)
            x = self._conv2(x)
            x = self._pool(x)
            x = self._conv3(x)
            x = self._conv4(x)
            x = self._pool(x)
            x = self._conv5(x)
            x = self._conv6(x)
            x = self._conv7(x)
            x = self._conv8(x)
        if self.num_classes > 0:
            x = self._drop(x)
            x = self._conv9(x)
        if self.with_pool:
            x = F.relu(x)
            x = self._avg_pool(x)
            x = paddle.squeeze(x, axis=[2, 3])

        return x


def _squeezenet(arch, version, pretrained, **kwargs):
    model = SqueezeNet(version, **kwargs)
    if pretrained:
        model = load_model_clas(model, arch, model_urls)
    return model


def squeezenet1_0(pretrained=False, **kwargs):
    """SqueezeNet v1.0 model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            from paddle.vision.models import squeezenet1_0

            # build model
            model = squeezenet1_0()

            # build model and load imagenet pretrained weight
            # model = squeezenet1_0(pretrained=True)
    """
    return _squeezenet('squeezenet1_0', '1.0', pretrained, **kwargs)


def squeezenet1_1(pretrained=False, **kwargs):
    """SqueezeNet v1.1 model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            from paddle.vision.models import squeezenet1_1

            # build model
            model = squeezenet1_1()

            # build model and load imagenet pretrained weight
            # model = squeezenet1_1(pretrained=True)
    """
    return _squeezenet('squeezenet1_1', '1.1', pretrained, **kwargs)
