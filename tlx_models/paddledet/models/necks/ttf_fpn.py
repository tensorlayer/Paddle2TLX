import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import xavier_uniform
from tensorlayerx.nn.initializers import Constant
from tensorlayerx.nn.initializers import random_uniform
from tensorlayerx.nn.initializers import random_normal
from tensorlayerx.nn.initializers import XavierUniform
from core.workspace import register
from core.workspace import serializable
from paddle.regularizer import L2Decay
from models.layers import DeformableConvV2
from models.layers import ConvNormLayer
from models.layers import LiteConv
import math
from models.ops import batch_norm
from ..shape_spec import ShapeSpec
__all__ = ['TTFFPN']


class Upsample(nn.Module):

    def __init__(self, ch_in, ch_out, norm_type='bn'):
        super(Upsample, self).__init__()
        fan_in = ch_in * 3 * 3
        stdv = 1.0 / math.sqrt(fan_in)
        self.dcn = DeformableConvV2(ch_in, ch_out, kernel_size=3, lr_scale=\
            2.0, regularizer=L2Decay(0.0))
        self.bn = batch_norm(ch_out, norm_type=norm_type, initializer=\
            Constant(1.0))

    def forward(self, feat):
        dcn = self.dcn(feat)
        bn = self.bn(dcn)
        relu = tensorlayerx.ops.relu(bn)
        out = paddle.nn.functional.interpolate(relu, scale_factor=2.0, mode
            ='bilinear')
        return out


class DeConv(nn.Module):

    def __init__(self, ch_in, ch_out, norm_type='bn'):
        super(DeConv, self).__init__()
        self.deconv = nn.Sequential()
        conv1 = ConvNormLayer(ch_in=ch_in, ch_out=ch_out, stride=1,
            filter_size=1, norm_type=norm_type, initializer=XavierUniform())
        conv2 = paddle2tlx.pd2tlx.ops.tlxops.tlx_ConvTranspose2d(in_channels
            =ch_out, out_channels=ch_out, kernel_size=4, padding=1, stride=\
            2, data_format='channels_first', b_init=False, W_init=\
            xavier_uniform(), n_group=ch_out)
        bn = batch_norm(ch_out, norm_type=norm_type, norm_decay=0.0)
        conv3 = ConvNormLayer(ch_in=ch_out, ch_out=ch_out, stride=1,
            filter_size=1, norm_type=norm_type, initializer=XavierUniform())
        self.deconv.add_sublayer('conv1', conv1)
        self.deconv.add_sublayer('relu6_1', nn.ReLU6())
        self.deconv.add_sublayer('conv2', conv2)
        self.deconv.add_sublayer('bn', bn)
        self.deconv.add_sublayer('relu6_2', nn.ReLU6())
        self.deconv.add_sublayer('conv3', conv3)
        self.deconv.add_sublayer('relu6_3', nn.ReLU6())

    def forward(self, inputs):
        return self.deconv(inputs)


class LiteUpsample(nn.Module):

    def __init__(self, ch_in, ch_out, norm_type='bn'):
        super(LiteUpsample, self).__init__()
        self.deconv = DeConv(ch_in, ch_out, norm_type=norm_type)
        self.conv = LiteConv(ch_in, ch_out, norm_type=norm_type)

    def forward(self, inputs):
        deconv_up = self.deconv(inputs)
        conv = self.conv(inputs)
        interp_up = paddle.nn.functional.interpolate(conv, scale_factor=2.0,
            mode='bilinear')
        return deconv_up + interp_up


class ShortCut(nn.Module):

    def __init__(self, layer_num, ch_in, ch_out, norm_type='bn', lite_neck=\
        False, name=None):
        super(ShortCut, self).__init__()
        _shortcut_conv = []
        for i in range(layer_num):
            fan_out = 3 * 3 * ch_out
            std = math.sqrt(2.0 / fan_out)
            in_channels = ch_in if i == 0 else ch_out
            shortcut_name = name + '.conv.{}'.format(i)
            if lite_neck:
                _shortcut_conv.append(LiteConv(in_channels=in_channels,
                    out_channels=ch_out, with_act=i < layer_num - 1,
                    norm_type=norm_type))
            else:
                _shortcut_conv.append(nn.GroupConv2d(in_channels=\
                    in_channels, out_channels=ch_out, kernel_size=3,
                    padding=1, W_init=xavier_uniform(), b_init=\
                    xavier_uniform(), data_format='channels_first'))
                if i < layer_num - 1:
                    _shortcut_conv.append(nn.ReLU())
        self.shortcut = nn.Sequential([*_shortcut_conv])

    def forward(self, feat):
        out = self.shortcut(feat)
        return out


@register
@serializable
class TTFFPN(nn.Module):
    """
    Args:
        in_channels (list): number of input feature channels from backbone.
            [128,256,512,1024] by default, means the channels of DarkNet53
            backbone return_idx [1,2,3,4].
        planes (list): the number of output feature channels of FPN.
            [256, 128, 64] by default
        shortcut_num (list): the number of convolution layers in each shortcut.
            [3,2,1] by default, means DarkNet53 backbone return_idx_1 has 3 convs
            in its shortcut, return_idx_2 has 2 convs and return_idx_3 has 1 conv.
        norm_type (string): norm type, 'sync_bn', 'bn', 'gn' are optional. 
            bn by default
        lite_neck (bool): whether to use lite conv in TTFNet FPN, 
            False by default
        fusion_method (string): the method to fusion upsample and lateral layer.
            'add' and 'concat' are optional, add by default
    """
    __shared__ = ['norm_type']

    def __init__(self, in_channels, planes=[256, 128, 64], shortcut_num=[3,
        2, 1], norm_type='bn', lite_neck=False, fusion_method='add'):
        super(TTFFPN, self).__init__()
        self.planes = planes
        self.shortcut_num = shortcut_num[::-1]
        self.shortcut_len = len(shortcut_num)
        self.ch_in = in_channels[::-1]
        self.fusion_method = fusion_method
        self.upsample_list = []
        self.shortcut_list = []
        self.upper_list = []
        for i, out_c in enumerate(self.planes):
            in_c = self.ch_in[i] if i == 0 else self.upper_list[-1]
            upsample_module = LiteUpsample if lite_neck else Upsample
            upsample = self.add_sublayer('upsample.' + str(i),
                upsample_module(in_c, out_c, norm_type=norm_type))
            self.upsample_list.append(upsample)
            if i < self.shortcut_len:
                shortcut = self.add_sublayer('shortcut.' + str(i), ShortCut
                    (self.shortcut_num[i], self.ch_in[i + 1], out_c,
                    norm_type=norm_type, lite_neck=lite_neck, name=\
                    'shortcut.' + str(i)))
                self.shortcut_list.append(shortcut)
                if self.fusion_method == 'add':
                    upper_c = out_c
                elif self.fusion_method == 'concat':
                    upper_c = out_c * 2
                else:
                    raise ValueError(
                        'Illegal fusion method. Expected add or                        concat, but received {}'
                        .format(self.fusion_method))
                self.upper_list.append(upper_c)

    def forward(self, inputs):
        feat = inputs[-1]
        for i, out_c in enumerate(self.planes):
            feat = self.upsample_list[i](feat)
            if i < self.shortcut_len:
                shortcut = self.shortcut_list[i](inputs[-i - 2])
                if self.fusion_method == 'add':
                    feat = feat + shortcut
                else:
                    feat = tensorlayerx.concat([feat, shortcut], axis=1)
        return feat

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape]}

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.upper_list[-1])]
