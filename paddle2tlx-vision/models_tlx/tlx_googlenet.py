# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import os
os.environ['TL_BACKEND'] = 'paddle'  # TODO - NCHW(pd) -> NHWC(tlx)
import paddle
# import paddle.nn as nn
import tensorlayerx.nn as nn
import tensorlayerx as tlx
# import paddle.nn.functional as F

# from paddle.nn import Conv2D, Linear, Dropout
from tensorlayerx.nn import Conv2d, Linear, Dropout, ReLU
# from paddle.nn import MaxPool2D, AvgPool2D, AdaptiveAvgPool2D
from tensorlayerx.nn import MaxPool2d, MeanPool2d, AvgPool2d, AdaptiveMeanPool2d, AdaptiveAvgPool2d
# from paddle.nn.initializer import Uniform
from tensorlayerx.nn.initializers import random_uniform
# from paddle.fluid.param_attr import ParamAttr
from utils.download import get_weights_path_from_url
from utils.load_model import restore_model

__all__ = []

model_urls = {
    "googlenet":
    ("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GoogLeNet_pretrained.pdparams",
     "80c06f038e905c53ab32c40eca6e26ae")
}


def xavier(channels, filter_size):
    stdv = (3.0 / (filter_size**2 * channels))**0.5
    # param_attr = ParamAttr(initializer=Uniform(-stdv, stdv))
    param_attr = random_uniform(-stdv, stdv)
    return param_attr


# class ConvLayer(nn.Layer):
class ConvLayer(nn.Module):

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1):
        super(ConvLayer, self).__init__()

        # self._conv = Conv2D(in_channels=num_channels,
        #                     out_channels=num_filters,
        #                     kernel_size=filter_size,
        #                     stride=stride,
        #                     padding=(filter_size - 1) // 2,
        #                     groups=groups,
        #                     bias_attr=False)
        self._conv = Conv2d(in_channels=num_channels,
                            out_channels=num_filters,
                            kernel_size=filter_size,
                            stride=stride,
                            padding=(filter_size - 1) // 2,
                            data_format='channels_first',
                            b_init=False)

    def forward(self, inputs):
        y = self._conv(inputs)
        return y


# class Inception(nn.Layer):
class Inception(nn.Module):

    def __init__(self, input_channels, output_channels, filter1, filter3R,
                 filter3, filter5R, filter5, proj):
        super(Inception, self).__init__()

        self._conv1 = ConvLayer(input_channels, filter1, 1)
        self._conv3r = ConvLayer(input_channels, filter3R, 1)
        self._conv3 = ConvLayer(filter3R, filter3, 3)
        self._conv5r = ConvLayer(input_channels, filter5R, 1)
        self._conv5 = ConvLayer(filter5R, filter5, 5)
        # self._pool = MaxPool2D(kernel_size=3, stride=1, padding=1)
        self._pool = MaxPool2d(kernel_size=3, stride=1, padding=1, data_format='channels_first')

        self._convprj = ConvLayer(input_channels, proj, 1)

    def forward(self, inputs):
        conv1 = self._conv1(inputs)

        conv3r = self._conv3r(inputs)
        conv3 = self._conv3(conv3r)

        conv5r = self._conv5r(inputs)
        conv5 = self._conv5(conv5r)

        pool = self._pool(inputs)
        convprj = self._convprj(pool)

        # cat = paddle.concat([conv1, conv3, conv5, convprj], axis=1)
        cat = tlx.concat([conv1, conv3, conv5, convprj], axis=1)
        # print('### cat shape:', cat.shape)
        # cat = F.relu(cat)
        cat = ReLU()(cat)
        return cat


# class GoogLeNet(nn.Layer):
class GoogLeNet(nn.Module):
    """GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <https://arxiv.org/pdf/1409.4842.pdf>`_
    
    Args:
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 1000.
        with_pool (bool, optional): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import GoogLeNet

            # build model
            model = GoogLeNet()

            x = paddle.rand([1, 3, 224, 224])
            out, out1, out2 = model(x)

            print(out.shape)
    """

    def __init__(self, num_classes=1000, with_pool=True):
        super(GoogLeNet, self).__init__()
        self.num_classes = num_classes
        self.with_pool = with_pool

        self._conv = ConvLayer(3, 64, 7, 2)
        # self._pool = MaxPool2D(kernel_size=3, stride=2)
        self._pool = MaxPool2d(kernel_size=3, stride=2, padding=0, data_format='channels_first')  # 0|VALID, 1|SAME
        self._conv_1 = ConvLayer(64, 64, 1)
        self._conv_2 = ConvLayer(64, 192, 3)

        self._ince3a = Inception(192, 192, 64, 96, 128, 16, 32, 32)
        self._ince3b = Inception(256, 256, 128, 128, 192, 32, 96, 64)

        self._ince4a = Inception(480, 480, 192, 96, 208, 16, 48, 64)
        self._ince4b = Inception(512, 512, 160, 112, 224, 24, 64, 64)
        self._ince4c = Inception(512, 512, 128, 128, 256, 24, 64, 64)
        self._ince4d = Inception(512, 512, 112, 144, 288, 32, 64, 64)
        self._ince4e = Inception(528, 528, 256, 160, 320, 32, 128, 128)

        self._ince5a = Inception(832, 832, 256, 160, 320, 32, 128, 128)
        self._ince5b = Inception(832, 832, 384, 192, 384, 48, 128, 128)

        if with_pool:
            # out
            # self._pool_5 = AdaptiveAvgPool2D(1)
            self._pool_5 = AdaptiveAvgPool2d(1, data_format='channels_first')
            # out1
            # self._pool_o1 = AvgPool2D(kernel_size=5, stride=3)
            self._pool_o1 = AvgPool2d(kernel_size=5, stride=3, padding=0, data_format='channels_first')
            # out2
            # self._pool_o2 = AvgPool2D(kernel_size=5, stride=3)
            self._pool_o2 = AvgPool2d(kernel_size=5, stride=3, padding=0, data_format='channels_first')

        if num_classes > 0:
            # out
            # self._drop = Dropout(p=0.4, mode="downscale_in_infer")
            self._drop = Dropout(p=0.4)
            # self._fc_out = Linear(1024,
            #                       num_classes,
            #                       weight_attr=xavier(1024, 1))
            self._fc_out = Linear(in_features=1024,
                                  out_features=num_classes,
                                  W_init=xavier(1024, 1)
                                  )

            # out1
            self._conv_o1 = ConvLayer(512, 128, 1)
            # self._fc_o1 = Linear(1152, 1024, weight_attr=xavier(2048, 1))
            self._fc_o1 = Linear(in_features=1152, out_features=1024, W_init=xavier(2048, 1))
            # self._drop_o1 = Dropout(p=0.7, mode="downscale_in_infer")
            self._drop_o1 = Dropout(p=0.7)
            # self._out1 = Linear(1024, num_classes, weight_attr=xavier(1024, 1))
            self._out1 = Linear(in_features=1024, out_features=num_classes, W_init=xavier(1024, 1))

            # out2
            self._conv_o2 = ConvLayer(528, 128, 1)
            # self._fc_o2 = Linear(1152, 1024, weight_attr=xavier(2048, 1))
            self._fc_o2 = Linear(in_features=1152, out_features=1024, W_init=xavier(2048, 1))
            # self._drop_o2 = Dropout(p=0.7, mode="downscale_in_infer")
            self._drop_o2 = Dropout(p=0.7)
            # self._out2 = Linear(1024, num_classes, weight_attr=xavier(1024, 1))
            self._out2 = Linear(in_features=1024, out_features=num_classes, W_init=xavier(1024, 1))

    def forward(self, inputs):
        x = self._conv(inputs)  # (1, 64, 112, 112)
        x = self._pool(x)  # (1, 64, 55, 55) <- (1, 64, 56, 56)
        x = self._conv_1(x)  # (1, 64, 55, 55)
        x = self._conv_2(x)  # (1, 192, 55, 55)
        x = self._pool(x)  # (1, 192, 27, 27)

        x = self._ince3a(x)  # (1, 256, 27, 27)
        x = self._ince3b(x)  # (1, 480, 27, 27)
        x = self._pool(x)  # (1, 480, 13, 13)

        ince4a = self._ince4a(x)  # (1, 512, 13, 13)
        x = self._ince4b(ince4a)  # (1, 512, 13, 13)
        x = self._ince4c(x)  # (1, 512, 13, 13)
        ince4d = self._ince4d(x)  # (1, 528, 13, 13)
        x = self._ince4e(ince4d)  # (1, 832, 13, 13)
        x = self._pool(x)  # (1, 832, 6, 6)

        x = self._ince5a(x)  # (1, 832, 6, 6)
        ince5b = self._ince5b(x)  # (1, 1024, 6, 6)

        out, out1, out2 = ince5b, ince4a, ince4d

        if self.with_pool:
            out = self._pool_5(out)  # (1, 1024, 1, 1)
            out1 = self._pool_o1(out1)  # (1, 512, 3, 3)
            out2 = self._pool_o2(out2)  # (1, 528, 3, 3)

        if self.num_classes > 0:
            out = self._drop(out)
            # out = paddle.squeeze(out, axis=[2, 3])
            out = tlx.ops.squeeze(out, axis=[2, 3])
            out = self._fc_out(out)  # (1, 1000)

            out1 = self._conv_o1(out1)  # (1, 128, 3, 3)
            # out1 = paddle.flatten(out1, start_axis=1, stop_axis=-1)  # (1, 1152)
            out1 = nn.Flatten()(out1)
            out1 = self._fc_o1(out1)  # (1, 1024)
            # out1 = F.relu(out1)
            out1 = ReLU()(out1)
            out1 = self._drop_o1(out1)
            out1 = self._out1(out1)  # (1, 1000)

            out2 = self._conv_o2(out2)  # (1, 128, 3, 3)
            # out2 = paddle.flatten(out2, start_axis=1, stop_axis=-1)  # (1, 1152)
            out2 = nn.Flatten()(out2)
            out2 = self._fc_o2(out2)
            out2 = self._drop_o2(out2)
            out2 = self._out2(out2)  # (1, 1000)
            # print(f"out.shape={out.shape}")
            # print(f"out1.shape={out1.shape}")
            # print(f"out2.shape={out2.shape}")
        return [out, out1, out2]


def googlenet(pretrained=False, **kwargs):
    """GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <https://arxiv.org/pdf/1409.4842.pdf>`_
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import googlenet

            # build model
            model = googlenet()

            # build model and load imagenet pretrained weight
            # model = googlenet(pretrained=True)

            x = paddle.rand([1, 3, 224, 224])
            out, out1, out2 = model(x)

            print(out.shape)
    """
    model = GoogLeNet(**kwargs)
    arch = "googlenet"
    if pretrained:
        assert (
            arch in model_urls
        ), "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        # model.set_dict(param)
        # model.load_dict(param)
        restore_model(param, model)
    return model
