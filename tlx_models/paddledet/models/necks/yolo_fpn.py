import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from core.workspace import register
from core.workspace import serializable
from models.layers import DropBlock
from models.ops import get_act_fn
from ..backbones.darknet import ConvBNLayer
from ..shape_spec import ShapeSpec
from ..backbones.csp_darknet import BaseConv
from ..backbones.csp_darknet import DWConv
from ..backbones.csp_darknet import CSPLayer
__all__ = ['YOLOv3FPN', 'YOLOCSPPAN']


def add_coord(x, data_format):
    b = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(x)[0]
    if data_format == 'NCHW':
        h, w = x.shape[2], x.shape[3]
    else:
        h, w = x.shape[1], x.shape[2]
    aa = tensorlayerx.ops.arange(w)
    bb = tensorlayerx.ops.arange(w)
    gx = tensorlayerx.cast(aa / ((w - 1.0) * 2.0) - 1.0, x.dtype)
    gy = tensorlayerx.cast(bb / ((h - 1.0) * 2.0) - 1.0, x.dtype)
    if data_format == 'NCHW':
        gx = gx.reshape([1, 1, 1, w]).expand([b, 1, h, w])
        gy = gy.reshape([1, 1, h, 1]).expand([b, 1, h, w])
    else:
        gx = gx.reshape([1, 1, w, 1]).expand([b, h, w, 1])
        gy = gy.reshape([1, h, 1, 1]).expand([b, h, w, 1])
    gx.stop_gradient = True
    gy.stop_gradient = True
    return gx, gy


class YoloDetBlock(nn.Module):

    def __init__(self, ch_in, channel, norm_type, freeze_norm=False, name=\
        '', data_format='channels_first'):
        """
        YOLODetBlock layer for yolov3, see https://arxiv.org/abs/1804.02767

        Args:
            ch_in (int): input channel
            channel (int): base channel
            norm_type (str): batch norm type
            freeze_norm (bool): whether to freeze norm, default False
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(YoloDetBlock, self).__init__()
        self.ch_in = ch_in
        self.channel = channel
        assert channel % 2 == 0, 'channel {} cannot be divided by 2'.format(
            channel)
        conv_def = [['conv0', ch_in, channel, 1, '.0.0'], ['conv1', channel,
            channel * 2, 3, '.0.1'], ['conv2', channel * 2, channel, 1,
            '.1.0'], ['conv3', channel, channel * 2, 3, '.1.1'], ['route', 
            channel * 2, channel, 1, '.2']]
        self.conv_module = nn.Sequential()
        for idx, (conv_name, ch_in, ch_out, filter_size, post_name
            ) in enumerate(conv_def):
            self.conv_module.add_sublayer(conv_name, ConvBNLayer(ch_in=\
                ch_in, ch_out=ch_out, filter_size=filter_size, padding=(
                filter_size - 1) // 2, norm_type=norm_type, freeze_norm=\
                freeze_norm, data_format=data_format, name=name + post_name))
        self.tip = ConvBNLayer(ch_in=channel, ch_out=channel * 2,
            filter_size=3, padding=1, norm_type=norm_type, freeze_norm=\
            freeze_norm, data_format=data_format, name=name + '.tip')

    def forward(self, inputs):
        route, tip = self.forward_tlx(inputs)
        return route, tip

    def forward_pd(self, x):
        route = self.conv_module(x)
        tip = self.tip(route)
        return route, tip

    def forward_tlx(self, x):
        for tlx_sub_layer in self.conv_module._sub_layers.values():
            x = tlx_sub_layer(x)
        tip = self.tip(x)
        return x, tip


class SPP(nn.Module):

    def __init__(self, ch_in, ch_out, k, pool_size, norm_type='bn',
        freeze_norm=False, name='', act='leaky', data_format='channels_first'):
        """
        SPP layer, which consist of four pooling layer follwed by conv layer

        Args:
            ch_in (int): input channel of conv layer
            ch_out (int): output channel of conv layer
            k (int): kernel size of conv layer
            norm_type (str): batch norm type
            freeze_norm (bool): whether to freeze norm, default False
            name (str): layer name
            act (str): activation function
            data_format (str): data format, NCHW or NHWC
        """
        super(SPP, self).__init__()
        self.pool = []
        self.data_format = data_format
        for size in pool_size:
            pool = self.add_sublayer('{}.pool1'.format(name), paddle2tlx.
                pd2tlx.ops.tlxops.tlx_MaxPool2d(kernel_size=size, stride=1,
                padding=size // 2, data_format=data_format, ceil_mode=False))
            self.pool.append(pool)
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, norm_type
            =norm_type, freeze_norm=freeze_norm, name=name, act=act,
            data_format=data_format)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        if self.data_format == 'NCHW':
            y = tensorlayerx.concat(outs, axis=1)
        else:
            y = tensorlayerx.concat(outs, axis=-1)
        y = self.conv(y)
        return y


class CoordConv(nn.Module):

    def __init__(self, ch_in, ch_out, filter_size, padding, norm_type,
        freeze_norm=False, name='', data_format='channels_first'):
        """
        CoordConv layer, see https://arxiv.org/abs/1807.03247

        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            filter_size (int): filter size, default 3
            padding (int): padding size, default 0
            norm_type (str): batch norm type, default bn
            name (str): layer name
            data_format (str): data format, NCHW or NHWC

        """
        super(CoordConv, self).__init__()
        self.conv = ConvBNLayer(ch_in + 2, ch_out, filter_size=filter_size,
            padding=padding, norm_type=norm_type, freeze_norm=freeze_norm,
            data_format=data_format, name=name)
        self.data_format = data_format

    def forward(self, x):
        gx, gy = add_coord(x, self.data_format)
        if self.data_format == 'NCHW' or self.data_format == 'channels_first':
            y = tensorlayerx.concat([x, gx, gy], axis=1)
        else:
            y = tensorlayerx.concat([x, gx, gy], axis=-1)
        y = self.conv(y)
        return y


class PPYOLODetBlock(nn.Module):

    def __init__(self, cfg, name, data_format='channels_first'):
        """
        PPYOLODetBlock layer

        Args:
            cfg (list): layer configs for this block
            name (str): block name
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLODetBlock, self).__init__()
        self.conv_module = nn.Sequential()
        for idx, (conv_name, layer, args, kwargs) in enumerate(cfg[:-1]):
            kwargs.update(name='{}.{}'.format(name, conv_name), data_format
                =data_format)
            self.conv_module.add_sublayer(conv_name, layer(*args, **kwargs))
        conv_name, layer, args, kwargs = cfg[-1]
        kwargs.update(name='{}.{}'.format(name, conv_name), data_format=\
            data_format)
        self.tip = layer(*args, **kwargs)

    def forward(self, inputs):
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip


class PPYOLOTinyDetBlock(nn.Module):

    def __init__(self, ch_in, ch_out, name, drop_block=False, block_size=3,
        keep_prob=0.9, data_format='channels_first'):
        """
        PPYOLO Tiny DetBlock layer
        Args:
            ch_in (list): input channel number
            ch_out (list): output channel number
            name (str): block name
            drop_block: whether user DropBlock
            block_size: drop block size
            keep_prob: probability to keep block in DropBlock
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLOTinyDetBlock, self).__init__()
        self.drop_block_ = drop_block
        self.conv_module = nn.Sequential()
        cfgs = [['.0', ch_in, ch_out, 1, 1, 0, 1], ['.1', ch_out, ch_out, 5,
            1, 2, ch_out], ['.2', ch_out, ch_out, 1, 1, 0, 1], ['.route',
            ch_out, ch_out, 5, 1, 2, ch_out]]
        for cfg in cfgs:
            (conv_name, conv_ch_in, conv_ch_out, filter_size, stride,
                padding, groups) = cfg
            self.conv_module.add_sublayer(name + conv_name, ConvBNLayer(
                ch_in=conv_ch_in, ch_out=conv_ch_out, filter_size=\
                filter_size, stride=stride, padding=padding, groups=groups,
                name=name + conv_name))
        self.tip = ConvBNLayer(ch_in=ch_out, ch_out=ch_out, filter_size=1,
            stride=1, padding=0, groups=1, name=name + conv_name)
        if self.drop_block_:
            self.drop_block = DropBlock(block_size=block_size, keep_prob=\
                keep_prob, data_format=data_format, name=name + '.dropblock')

    def forward(self, inputs):
        if self.drop_block_:
            inputs = self.drop_block(inputs)
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip


class PPYOLODetBlockCSP(nn.Module):

    def __init__(self, cfg, ch_in, ch_out, act, norm_type, name,
        data_format='channels_first'):
        """
        PPYOLODetBlockCSP layer

        Args:
            cfg (list): layer configs for this block
            ch_in (int): input channel
            ch_out (int): output channel
            act (str): default mish
            name (str): block name
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLODetBlockCSP, self).__init__()
        self.data_format = data_format
        self.conv1 = ConvBNLayer(ch_in, ch_out, 1, padding=0, act=act,
            norm_type=norm_type, name=name + '.left', data_format=data_format)
        self.conv2 = ConvBNLayer(ch_in, ch_out, 1, padding=0, act=act,
            norm_type=norm_type, name=name + '.right', data_format=data_format)
        self.conv3 = ConvBNLayer(ch_out * 2, ch_out * 2, 1, padding=0, act=\
            act, norm_type=norm_type, name=name, data_format=data_format)
        self.conv_module = nn.Sequential()
        for idx, (layer_name, layer, args, kwargs) in enumerate(cfg):
            kwargs.update(name=name + layer_name, data_format=data_format)
            self.conv_module.add_sublayer(layer_name, layer(*args, **kwargs))

    def forward(self, inputs):
        conv_left = self.conv1(inputs)
        conv_right = self.conv2(inputs)
        conv_left = self.conv_module(conv_left)
        if self.data_format == 'NCHW' or self.data_format == 'channels_first':
            conv = tensorlayerx.concat([conv_left, conv_right], axis=1)
        else:
            conv = tensorlayerx.concat([conv_left, conv_right], axis=-1)
        conv = self.conv3(conv)
        return conv, conv


@register
@serializable
class YOLOv3FPN(nn.Module):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self, in_channels=[256, 512, 1024], norm_type='bn',
        freeze_norm=False, data_format='channels_first'):
        """
        YOLOv3FPN layer

        Args:
            in_channels (list): input channels for fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC

        """
        super(YOLOv3FPN, self).__init__()
        assert len(in_channels) > 0, 'in_channels length should > 0'
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)
        self._out_channels = []
        self.yolo_blocks = []
        self.routes = []
        self.data_format = data_format
        for i in range(self.num_blocks):
            name = 'yolo_block.{}'.format(i)
            in_channel = in_channels[-i - 1]
            if i > 0:
                in_channel += 512 // 2 ** i
            yolo_block = self.add_sublayer(name, YoloDetBlock(in_channel,
                channel=512 // 2 ** i, norm_type=norm_type, freeze_norm=\
                freeze_norm, data_format=data_format, name=name))
            self.yolo_blocks.append(yolo_block)
            self._out_channels.append(1024 // 2 ** i)
            if i < self.num_blocks - 1:
                name = 'yolo_transition.{}'.format(i)
                route = self.add_sublayer(name, ConvBNLayer(ch_in=512 // 2 **
                    i, ch_out=256 // 2 ** i, filter_size=1, stride=1,
                    padding=0, norm_type=norm_type, freeze_norm=freeze_norm,
                    data_format=data_format, name=name))
                self.routes.append(route)

    def forward(self, blocks, for_mot=False):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        yolo_feats = []
        if for_mot:
            emb_feats = []
        for i, block in enumerate(blocks):
            if i > 0:
                if (self.data_format == 'NCHW' or self.data_format ==\
                    'channels_first'):
                    block = tensorlayerx.concat([route, block], axis=1)
                else:
                    block = tensorlayerx.concat([route, block], axis=-1)
            route, tip = self.yolo_blocks[i](block)
            yolo_feats.append(tip)
            if for_mot:
                emb_feats.append(route)
            if i < self.num_blocks - 1:
                route = self.routes[i](route)
                route = paddle.nn.functional.interpolate(route,
                    scale_factor=2.0)
        if for_mot:
            return {'yolo_feats': yolo_feats, 'emb_feats': emb_feats}
        else:
            return yolo_feats

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape]}

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]


@register
@serializable
class YOLOCSPPAN(nn.Module):
    """
    YOLO CSP-PAN, used in YOLOv5 and YOLOX.
    """
    __shared__ = ['depth_mult', 'data_format', 'act', 'trt']

    def __init__(self, depth_mult=1.0, in_channels=[256, 512, 1024],
        depthwise=False, data_format='channels_first', act='silu', trt=False):
        super(YOLOCSPPAN, self).__init__()
        self.in_channels = in_channels
        self._out_channels = in_channels
        Conv = DWConv if depthwise else BaseConv
        self.data_format = data_format
        act = get_act_fn(act, trt=trt) if act is None or isinstance(act, (
            str, dict)) else act
        self.upsample = paddle2tlx.pd2tlx.ops.tlxops.tlx_Upsample(scale_factor
            =2, mode='nearest', data_format='channels_first')
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(BaseConv(int(in_channels[idx]), int(
                in_channels[idx - 1]), 1, 1, act=act))
            self.fpn_blocks.append(CSPLayer(int(in_channels[idx - 1] * 2),
                int(in_channels[idx - 1]), round(3 * depth_mult), shortcut=\
                False, depthwise=depthwise, act=act))
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(Conv(int(in_channels[idx]), int(
                in_channels[idx]), 3, stride=2, act=act))
            self.pan_blocks.append(CSPLayer(int(in_channels[idx] * 2), int(
                in_channels[idx + 1]), round(3 * depth_mult), shortcut=\
                False, depthwise=depthwise, act=act))

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        inner_outs = [feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = paddle.nn.functional.interpolate(feat_heigh,
                scale_factor=2.0, mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                tensorlayerx.concat([upsample_feat, feat_low], axis=1))
            inner_outs.insert(0, inner_out)
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](tensorlayerx.concat([downsample_feat,
                feat_height], axis=1))
            outs.append(out)
        return outs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape]}

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
