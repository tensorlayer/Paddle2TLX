import tensorlayerx as tlx
import paddle
import paddle2tlx
import math
from numbers import Integral
import tensorlayerx
import tensorlayerx.nn as nn
from core.workspace import register
from core.workspace import serializable
from paddle.regularizer import L2Decay
from tensorlayerx.nn.initializers import random_uniform
from tensorlayerx.nn.initializers import xavier_uniform
from tensorlayerx.nn.initializers import Constant
from paddle2tlx.pd2tlx.ops.tlxops import tlx_DeformConv2d
from .name_adapter import NameAdapter
from ..shape_spec import ShapeSpec
from collections import OrderedDict
__all__ = ['ResNet', 'Res5Head', 'Blocks', 'BasicBlock', 'BottleNeck']
ResNet_cfg = {(18): [2, 2, 2, 2], (34): [3, 4, 6, 3], (50): [3, 4, 6, 3], (
    101): [3, 4, 23, 3], (152): [3, 8, 36, 3]}


class ConvNormLayer(nn.Module):

    def __init__(self, ch_in, ch_out, filter_size, stride, groups=1, act=\
        None, norm_type='bn', norm_decay=0.0, freeze_norm=True, lr=1.0,
        dcn_v2=False):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn']
        self.norm_type = norm_type
        self.act = act
        self.dcn_v2 = dcn_v2
        if not self.dcn_v2:
            self.conv = nn.GroupConv2d(in_channels=ch_in, out_channels=\
                ch_out, kernel_size=filter_size, stride=stride, padding=(
                filter_size - 1) // 2, W_init=xavier_uniform(), b_init=\
                False, n_group=groups, data_format='channels_first')
        else:
            self.offset_channel = 2 * filter_size ** 2
            self.mask_channel = filter_size ** 2
            self.conv_offset = nn.GroupConv2d(in_channels=ch_in,
                out_channels=3 * filter_size ** 2, kernel_size=filter_size,
                stride=stride, padding=(filter_size - 1) // 2, W_init=\
                xavier_uniform(), b_init=xavier_uniform(), data_format=\
                'channels_first')
            self.conv = tlx_DeformConv2d(in_channels=ch_in, out_channels=\
                ch_out, kernel_size=filter_size, stride=stride, padding=(
                filter_size - 1) // 2, dilation=1, groups=groups, W_init=\
                xavier_uniform(), b_init=None)
        norm_lr = 0.0 if freeze_norm else lr
        param_attr = xavier_uniform()
        bias_attr = xavier_uniform()
        global_stats = True if freeze_norm else None
        if norm_type in ['sync_bn', 'bn']:
            self.norm = nn.BatchNorm2d(num_features=ch_out, data_format=\
                'channels_first')
        norm_params = self.norm.parameters()
        if freeze_norm:
            for param in norm_params:
                param.stop_gradient = True

    def forward(self, inputs):
        if not self.dcn_v2:
            out = self.conv(inputs)
        else:
            offset_mask = self.conv_offset(inputs)
            offset, mask = tensorlayerx.ops.split(offset_mask, axis=1,
                num_or_size_splits=[self.offset_channel, self.mask_channel])
            mask = tensorlayerx.ops.sigmoid(mask)
            out = self.conv(inputs, offset, mask=mask)
        if self.norm_type in ['bn', 'sync_bn']:
            out = self.norm(out)
        if self.act:
            out = tensorlayerx.ops.relu(out)
        return out


class SELayer(nn.Module):

    def __init__(self, ch, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1, data_format='channels_first')
        stdv = 1.0 / math.sqrt(ch)
        c_ = ch // reduction_ratio
        self.squeeze = nn.Linear(in_features=ch, out_features=c_, W_init=\
            tensorlayerx.initializers.random_uniform(-stdv, stdv), b_init=True)
        stdv = 1.0 / math.sqrt(c_)
        self.extract = nn.Linear(in_features=c_, out_features=ch, W_init=\
            tensorlayerx.initializers.random_uniform(-stdv, stdv), b_init=True)

    def forward(self, inputs):
        out = self.pool(inputs)
        out = tensorlayerx.ops.squeeze(out, axis=[2, 3])
        out = self.squeeze(out)
        out = tensorlayerx.ops.relu(out)
        out = self.extract(out)
        out = tensorlayerx.ops.sigmoid(out)
        out = tensorlayerx.expand_dims(out, axis=[2, 3])
        scale = out * inputs
        return scale


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, variant='b', groups
        =1, base_width=64, lr=1.0, norm_type='bn', norm_decay=0.0,
        freeze_norm=True, dcn_v2=False, std_senet=False):
        super(BasicBlock, self).__init__()
        assert groups == 1 and base_width == 64, 'BasicBlock only supports groups=1 and base_width=64'
        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([('pool', paddle2tlx
                    .pd2tlx.ops.tlxops.tlx_AvgPool2d(kernel_size=2, stride=\
                    2, padding=0, ceil_mode=True)), ('conv', ConvNormLayer(
                    ch_in=ch_in, ch_out=ch_out, filter_size=1, stride=1,
                    norm_type=norm_type, norm_decay=norm_decay, freeze_norm
                    =freeze_norm, lr=lr))]))
            else:
                self.short = ConvNormLayer(ch_in=ch_in, ch_out=ch_out,
                    filter_size=1, stride=stride, norm_type=norm_type,
                    norm_decay=norm_decay, freeze_norm=freeze_norm, lr=lr)
        self.branch2a = ConvNormLayer(ch_in=ch_in, ch_out=ch_out,
            filter_size=3, stride=stride, act='relu', norm_type=norm_type,
            norm_decay=norm_decay, freeze_norm=freeze_norm, lr=lr)
        self.branch2b = ConvNormLayer(ch_in=ch_out, ch_out=ch_out,
            filter_size=3, stride=1, act=None, norm_type=norm_type,
            norm_decay=norm_decay, freeze_norm=freeze_norm, lr=lr, dcn_v2=\
            dcn_v2)
        self.std_senet = std_senet
        if self.std_senet:
            self.se = SELayer(ch_out)

    def forward(self, inputs):
        out = self.branch2a(inputs)
        out = self.branch2b(out)
        if self.std_senet:
            out = self.se(out)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        out = tensorlayerx.add(value=out, bias=short)
        out = tensorlayerx.ops.relu(out)
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, variant='b', groups
        =1, base_width=4, lr=1.0, norm_type='bn', norm_decay=0.0,
        freeze_norm=True, dcn_v2=False, std_senet=False):
        super(BottleNeck, self).__init__()
        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride
        width = int(ch_out * (base_width / 64.0)) * groups
        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([('pool', paddle2tlx
                    .pd2tlx.ops.tlxops.tlx_AvgPool2d(kernel_size=2, stride=\
                    2, padding=0, ceil_mode=True)), ('conv', ConvNormLayer(
                    ch_in=ch_in, ch_out=ch_out * self.expansion,
                    filter_size=1, stride=1, norm_type=norm_type,
                    norm_decay=norm_decay, freeze_norm=freeze_norm, lr=lr))]))
            else:
                self.short = ConvNormLayer(ch_in=ch_in, ch_out=ch_out *
                    self.expansion, filter_size=1, stride=stride, norm_type
                    =norm_type, norm_decay=norm_decay, freeze_norm=\
                    freeze_norm, lr=lr)
        self.branch2a = ConvNormLayer(ch_in=ch_in, ch_out=width,
            filter_size=1, stride=stride1, groups=1, act='relu', norm_type=\
            norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm, lr=lr)
        self.branch2b = ConvNormLayer(ch_in=width, ch_out=width,
            filter_size=3, stride=stride2, groups=groups, act='relu',
            norm_type=norm_type, norm_decay=norm_decay, freeze_norm=\
            freeze_norm, lr=lr, dcn_v2=dcn_v2)
        self.branch2c = ConvNormLayer(ch_in=width, ch_out=ch_out * self.
            expansion, filter_size=1, stride=1, groups=1, norm_type=\
            norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm, lr=lr)
        self.std_senet = std_senet
        if self.std_senet:
            self.se = SELayer(ch_out * self.expansion)

    def forward(self, inputs):
        out = self.branch2a(inputs)
        out = self.branch2b(out)
        out = self.branch2c(out)
        if self.std_senet:
            out = self.se(out)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        out = tensorlayerx.add(value=out, bias=short)
        out = tensorlayerx.ops.relu(out)
        return out


class Blocks(nn.Module):

    def __init__(self, block, ch_in, ch_out, count, name_adapter, stage_num,
        variant='b', groups=1, base_width=64, lr=1.0, norm_type='bn',
        norm_decay=0.0, freeze_norm=True, dcn_v2=False, std_senet=False):
        super(Blocks, self).__init__()
        self.blocks = []
        for i in range(count):
            conv_name = name_adapter.fix_layer_warp_name(stage_num, count, i)
            layer = self.add_sublayer(conv_name, block(ch_in=ch_in, ch_out=\
                ch_out, stride=2 if i == 0 and stage_num != 2 else 1,
                shortcut=False if i == 0 else True, variant=variant, groups
                =groups, base_width=base_width, lr=lr, norm_type=norm_type,
                norm_decay=norm_decay, freeze_norm=freeze_norm, dcn_v2=\
                dcn_v2, std_senet=std_senet))
            self.blocks.append(layer)
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, inputs):
        block_out = inputs
        for block in self.blocks:
            block_out = block(block_out)
        return block_out


@register
@serializable
class ResNet(nn.Module):
    __shared__ = ['norm_type']

    def __init__(self, depth=50, ch_in=64, variant='b', lr_mult_list=[1.0, 
        1.0, 1.0, 1.0], groups=1, base_width=64, norm_type='bn', norm_decay
        =0, freeze_norm=True, freeze_at=0, return_idx=[0, 1, 2, 3],
        dcn_v2_stages=[-1], num_stages=4, std_senet=False):
        """
        Residual Network, see https://arxiv.org/abs/1512.03385
        
        Args:
            depth (int): ResNet depth, should be 18, 34, 50, 101, 152.
            ch_in (int): output channel of first stage, default 64
            variant (str): ResNet variant, supports 'a', 'b', 'c', 'd' currently
            lr_mult_list (list): learning rate ratio of different resnet stages(2,3,4,5),
                                 lower learning rate ratio is need for pretrained model 
                                 got using distillation(default as [1.0, 1.0, 1.0, 1.0]).
            groups (int): group convolution cardinality
            base_width (int): base width of each group convolution
            norm_type (str): normalization type, 'bn', 'sync_bn' or 'affine_channel'
            norm_decay (float): weight decay for normalization layer weights
            freeze_norm (bool): freeze normalization layers
            freeze_at (int): freeze the backbone at which stage
            return_idx (list): index of the stages whose feature maps are returned
            dcn_v2_stages (list): index of stages who select deformable conv v2
            num_stages (int): total num of stages
            std_senet (bool): whether use senet, default True
        """
        super(ResNet, self).__init__()
        self._model_type = 'ResNet' if groups == 1 else 'ResNeXt'
        assert num_stages >= 1 and num_stages <= 4
        self.depth = depth
        self.variant = variant
        self.groups = groups
        self.base_width = base_width
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self.freeze_at = freeze_at
        if isinstance(return_idx, Integral):
            return_idx = [return_idx]
        assert max(return_idx
            ) < num_stages, 'the maximum return index must smaller than num_stages, but received maximum return index is {} and num_stages is {}'.format(
            max(return_idx), num_stages)
        self.return_idx = return_idx
        self.num_stages = num_stages
        assert len(lr_mult_list
            ) == 4, 'lr_mult_list length must be 4 but got {}'.format(len(
            lr_mult_list))
        if isinstance(dcn_v2_stages, Integral):
            dcn_v2_stages = [dcn_v2_stages]
        assert max(dcn_v2_stages) < num_stages
        if isinstance(dcn_v2_stages, Integral):
            dcn_v2_stages = [dcn_v2_stages]
        assert max(dcn_v2_stages) < num_stages
        self.dcn_v2_stages = dcn_v2_stages
        block_nums = ResNet_cfg[depth]
        na = NameAdapter(self)
        conv1_name = na.fix_c1_stage_name()
        if variant in ['c', 'd']:
            conv_def = [[3, ch_in // 2, 3, 2, 'conv1_1'], [ch_in // 2, 
                ch_in // 2, 3, 1, 'conv1_2'], [ch_in // 2, ch_in, 3, 1,
                'conv1_3']]
        else:
            conv_def = [[3, ch_in, 7, 2, conv1_name]]
        self.conv1 = nn.Sequential()
        for c_in, c_out, k, s, _name in conv_def:
            self.conv1.add_sublayer(_name, ConvNormLayer(ch_in=c_in, ch_out
                =c_out, filter_size=k, stride=s, groups=1, act='relu',
                norm_type=norm_type, norm_decay=norm_decay, freeze_norm=\
                freeze_norm, lr=1.0))
        self.ch_in = ch_in
        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock
        self._out_channels = [(block.expansion * v) for v in ch_out_list]
        self._out_strides = [4, 8, 16, 32]
        self.res_layers = []
        for i in range(num_stages):
            lr_mult = lr_mult_list[i]
            stage_num = i + 2
            res_name = 'res{}'.format(stage_num)
            res_layer = self.add_sublayer(res_name, Blocks(block, self.
                ch_in, ch_out_list[i], count=block_nums[i], name_adapter=na,
                stage_num=stage_num, variant=variant, groups=groups,
                base_width=base_width, lr=lr_mult, norm_type=norm_type,
                norm_decay=norm_decay, freeze_norm=freeze_norm, dcn_v2=i in
                self.dcn_v2_stages, std_senet=std_senet))
            self.res_layers.append(res_layer)
            self.ch_in = self._out_channels[i]
        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at + 1, num_stages)):
                self._freeze_parameters(self.res_layers[i])

    def _freeze_parameters(self, m):
        for p in m.parameters():
            p.stop_gradient = True

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self._out_channels[i], stride=self.
            _out_strides[i]) for i in self.return_idx]

    def forward(self, inputs):
        x = inputs['image']
        print(f'x={x}')
        conv1 = self.conv1(x)
        x = paddle.nn.functional.max_pool2d(conv1, kernel_size=3, stride=2,
            padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


@register
class Res5Head(nn.Module):

    def __init__(self, depth=50):
        super(Res5Head, self).__init__()
        feat_in, feat_out = [1024, 512]
        if depth < 50:
            feat_in = 256
        na = NameAdapter(self)
        block = BottleNeck if depth >= 50 else BasicBlock
        self.res5 = Blocks(block, feat_in, feat_out, count=3, name_adapter=\
            na, stage_num=5)
        self.feat_out = feat_out if depth < 50 else feat_out * 4

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.feat_out, stride=16)]

    def forward(self, roi_feat, stage=0):
        y = self.res5(roi_feat)
        return y
