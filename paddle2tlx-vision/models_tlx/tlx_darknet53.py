# from __future__ import absolute_import
# from __future__ import division
import os
os.environ['TL_BACKEND'] = 'paddle'

import tensorlayerx as tlx
import tensorlayerx.nn as nn

from paddle.utils.download import get_weights_path_from_url
import paddle
from utils.load_model import restore_model
import math


MODEL_URLS = {
    "DarkNet53":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DarkNet53_pretrained.pdparams"
}

__all__ = []

class ConvBNLayer(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 filter_size,
                 stride,
                 padding):
        super(ConvBNLayer, self).__init__()
        self._conv = tlx.nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            data_format='channels_first',
            W_init=tlx.initializers.xavier_uniform(),
            b_init=None,
            )
        # self._conv = paddle.nn.Conv2D(
        #     in_channels=input_channels,
        #     out_channels=output_channels,
        #     kernel_size=filter_size,
        #     stride=stride,
        #     padding=padding,
        #     weight_attr=paddle.ParamAttr(),
        #     bias_attr=False)

        self._bn = tlx.nn.BatchNorm(
            num_features=output_channels,
            act="relu",
            data_format='channels_first',
            moving_mean_init=tlx.initializers.xavier_uniform(),
            moving_var_init=tlx.initializers.xavier_uniform()
            )
        # self._bn = paddle.nn.BatchNorm(
        #     num_channels=output_channels,
        #     act="relu",
        #     param_attr=paddle.ParamAttr(),
        #     bias_attr=paddle.ParamAttr(),
        #     )

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._bn(x)
        return x
    
    
class BasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(BasicBlock, self).__init__()

        self._conv1 = ConvBNLayer(
            input_channels, output_channels, 1, 1, 0)
        self._conv2 = ConvBNLayer(
            output_channels, output_channels * 2, 3, 1, 1)

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)
        return tlx.ops.add(inputs, x)


class DarkNet(nn.Module):
    def __init__(self, class_num=1000):
        super(DarkNet, self).__init__()

        self.stages = [1, 2, 8, 8, 4]
        self._conv1 = ConvBNLayer(3, 32, 3, 1, 1)
        self._conv2 = ConvBNLayer(
            32, 64, 3, 2, 1)

        self._basic_block_01 = BasicBlock(64, 32)
        self._downsample_0 = ConvBNLayer(
            64, 128, 3, 2, 1)

        self._basic_block_11 = BasicBlock(128, 64)
        self._basic_block_12 = BasicBlock(128, 64)
        self._downsample_1 = ConvBNLayer(
            128, 256, 3, 2, 1)

        self._basic_block_21 = BasicBlock(256, 128)
        self._basic_block_22 = BasicBlock(256, 128)
        self._basic_block_23 = BasicBlock(256, 128)
        self._basic_block_24 = BasicBlock(256, 128)
        self._basic_block_25 = BasicBlock(256, 128)
        self._basic_block_26 = BasicBlock(256, 128)
        self._basic_block_27 = BasicBlock(256, 128)
        self._basic_block_28 = BasicBlock(256, 128)
        self._downsample_2 = ConvBNLayer(
            256, 512, 3, 2, 1)

        self._basic_block_31 = BasicBlock(512, 256)
        self._basic_block_32 = BasicBlock(512, 256)
        self._basic_block_33 = BasicBlock(512, 256)
        self._basic_block_34 = BasicBlock(512, 256)
        self._basic_block_35 = BasicBlock(512, 256)
        self._basic_block_36 = BasicBlock(512, 256)
        self._basic_block_37 = BasicBlock(512, 256)
        self._basic_block_38 = BasicBlock(512, 256)
        self._downsample_3 = ConvBNLayer(
            512, 1024, 3, 2, 1)

        self._basic_block_41 = BasicBlock(1024, 512)
        self._basic_block_42 = BasicBlock(1024, 512)
        self._basic_block_43 = BasicBlock(1024, 512)
        self._basic_block_44 = BasicBlock(1024, 512)

        self._pool = tlx.nn.AdaptiveAvgPool2d(1, data_format='channels_first')

        stdv = 1.0 / math.sqrt(1024.0)
        self._out = tlx.nn.Linear(
            in_features=1024,
            out_features =class_num,
            W_init=tlx.initializers.random_uniform(minval=-0.05, maxval=0.05),
            b_init=tlx.initializers.xavier_uniform())

    def forward(self, inputs):
        x = self._conv1(inputs)
        # print('tlx x 0: ', x)
        x = self._conv2(x)

        x = self._basic_block_01(x)
        x = self._downsample_0(x)

        x = self._basic_block_11(x)
        x = self._basic_block_12(x)
        x = self._downsample_1(x)

        x = self._basic_block_21(x)
        x = self._basic_block_22(x)
        x = self._basic_block_23(x)
        x = self._basic_block_24(x)
        x = self._basic_block_25(x)
        x = self._basic_block_26(x)
        x = self._basic_block_27(x)
        x = self._basic_block_28(x)
        x = self._downsample_2(x)

        x = self._basic_block_31(x)
        x = self._basic_block_32(x)
        x = self._basic_block_33(x)
        x = self._basic_block_34(x)
        x = self._basic_block_35(x)
        x = self._basic_block_36(x)
        x = self._basic_block_37(x)
        x = self._basic_block_38(x)
        x = self._downsample_3(x)

        x = self._basic_block_41(x)
        x = self._basic_block_42(x)
        x = self._basic_block_43(x)
        x = self._basic_block_44(x)  # [1, 1024, 7, 7]

        x = self._pool(x)  # [1, 1, 1, 7]
        x = tlx.squeeze(x, axis=[2, 3])
        x = self._out(x)
        return x


def _darknet53(arch, pretrained, **kwargs):
    model = DarkNet(**kwargs)

    if pretrained:
        assert arch in MODEL_URLS, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(MODEL_URLS[arch])

        param = paddle.load(weight_path)
        restore_model(param, model)

    return model


def darknet53(pretrained=False, **kwargs):
    return _darknet53('DarkNet53', pretrained, **kwargs)
