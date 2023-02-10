from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import xavier_uniform
from tensorlayerx.nn import AdaptiveAvgPool2d
from tensorlayerx.nn import GroupConv2d
from paddle.regularizer import L2Decay
from tensorlayerx.nn.initializers import HeNormal
from core.workspace import register
from core.workspace import serializable
from numbers import Integral
from ..shape_spec import ShapeSpec
__all__ = ['LCNet']
NET_CONFIG = {'blocks2': [[3, 16, 32, 1, False]], 'blocks3': [[3, 32, 64, 2,
    False], [3, 64, 64, 1, False]], 'blocks4': [[3, 64, 128, 2, False], [3,
    128, 128, 1, False]], 'blocks5': [[3, 128, 256, 2, False], [5, 256, 256,
    1, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False], [5, 256, 
    256, 1, False], [5, 256, 256, 1, False]], 'blocks6': [[5, 256, 512, 2,
    True], [5, 512, 512, 1, True]]}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):

    def __init__(self, num_channels, filter_size, num_filters, stride,
        num_groups=1, act='hard_swish'):
        super().__init__()
        self.conv = GroupConv2d(in_channels=num_channels, out_channels=\
            num_filters, kernel_size=filter_size, stride=stride, padding=(
            filter_size - 1) // 2, W_init=xavier_uniform(), b_init=False,
            n_group=num_groups, data_format='channels_first')
        self.bn = nn.BatchNorm2d(num_features=num_filters, data_format=\
            'channels_first')
        if act == 'hard_swish':
            self.act = nn.Hardswish()
        elif act == 'relu6':
            self.act = nn.ReLU6()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DepthwiseSeparable(nn.Module):

    def __init__(self, num_channels, num_filters, stride, dw_size=3, use_se
        =False, act='hard_swish'):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(num_channels=num_channels, num_filters=\
            num_channels, filter_size=dw_size, stride=stride, num_groups=\
            num_channels, act=act)
        if use_se:
            self.se = SEModule(num_channels)
        self.pw_conv = ConvBNLayer(num_channels=num_channels, filter_size=1,
            num_filters=num_filters, stride=1, act=act)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class SEModule(nn.Module):

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1, data_format='channels_first')
        self.conv1 = GroupConv2d(in_channels=channel, out_channels=channel //
            reduction, kernel_size=1, stride=1, padding=0, data_format=\
            'channels_first')
        self.relu = nn.ReLU()
        self.conv2 = GroupConv2d(in_channels=channel // reduction,
            out_channels=channel, kernel_size=1, stride=1, padding=0,
            data_format='channels_first')
        self.hardsigmoid = nn.HardSigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = tensorlayerx.ops.multiply(x=identity, y=x)
        return x


@register
@serializable
class LCNet(nn.Module):

    def __init__(self, scale=1.0, feature_maps=[3, 4, 5], act='hard_swish'):
        super().__init__()
        self.scale = scale
        self.feature_maps = feature_maps
        out_channels = []
        self.conv1 = ConvBNLayer(num_channels=3, filter_size=3, num_filters
            =make_divisible(16 * scale), stride=2, act=act)
        self.blocks2 = self.blocks2_layer_tlx(scale, act)
        self.blocks3 = nn.Sequential([*[DepthwiseSeparable(num_channels=\
            make_divisible(in_c * scale), num_filters=make_divisible(out_c *
            scale), dw_size=k, stride=s, use_se=se, act=act) for i, (k,
            in_c, out_c, s, se) in enumerate(NET_CONFIG['blocks3'])]])
        out_channels.append(make_divisible(NET_CONFIG['blocks3'][-1][2] *
            scale))
        self.blocks4 = nn.Sequential([*[DepthwiseSeparable(num_channels=\
            make_divisible(in_c * scale), num_filters=make_divisible(out_c *
            scale), dw_size=k, stride=s, use_se=se, act=act) for i, (k,
            in_c, out_c, s, se) in enumerate(NET_CONFIG['blocks4'])]])
        out_channels.append(make_divisible(NET_CONFIG['blocks4'][-1][2] *
            scale))
        self.blocks5 = nn.Sequential([*[DepthwiseSeparable(num_channels=\
            make_divisible(in_c * scale), num_filters=make_divisible(out_c *
            scale), dw_size=k, stride=s, use_se=se, act=act) for i, (k,
            in_c, out_c, s, se) in enumerate(NET_CONFIG['blocks5'])]])
        out_channels.append(make_divisible(NET_CONFIG['blocks5'][-1][2] *
            scale))
        self.blocks6 = nn.Sequential([*[DepthwiseSeparable(num_channels=\
            make_divisible(in_c * scale), num_filters=make_divisible(out_c *
            scale), dw_size=k, stride=s, use_se=se, act=act) for i, (k,
            in_c, out_c, s, se) in enumerate(NET_CONFIG['blocks6'])]])
        out_channels.append(make_divisible(NET_CONFIG['blocks6'][-1][2] *
            scale))
        self._out_channels = [ch for idx, ch in enumerate(out_channels) if 
            idx + 2 in feature_maps]

    def blocks2_layer_pd(self, scale, act):
        blocks2 = nn.Sequential([*[DepthwiseSeparable(num_channels=\
            make_divisible(in_c * scale), num_filters=make_divisible(out_c *
            scale), dw_size=k, stride=s, use_se=se, act=act) for i, (k,
            in_c, out_c, s, se) in enumerate(NET_CONFIG['blocks2'])]])
        return blocks2

    def blocks2_layer_tlx(self, scale, act):
        if len(NET_CONFIG['blocks2']) == 1:
            blocks2 = nn.Sequential([DepthwiseSeparable(num_channels=\
                make_divisible(in_c * scale), num_filters=make_divisible(
                out_c * scale), dw_size=k, stride=s, use_se=se, act=act) for
                i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG['blocks2'])]
                )
        else:
            blocks2 = nn.Sequential([*[DepthwiseSeparable(num_channels=\
                make_divisible(in_c * scale), num_filters=make_divisible(
                out_c * scale), dw_size=k, stride=s, use_se=se, act=act) for
                i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG[
                'blocks2'])]])
        return blocks2

    def forward(self, inputs):
        x = inputs['image']
        outs = []
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        outs.append(x)
        x = self.blocks4(x)
        outs.append(x)
        x = self.blocks5(x)
        outs.append(x)
        x = self.blocks6(x)
        outs.append(x)
        outs = [o for i, o in enumerate(outs) if i + 2 in self.feature_maps]
        return outs

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
