import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import xavier_uniform
from tensorlayerx.nn.initializers import Constant
from tensorlayerx.nn.initializers import random_normal
from paddle.regularizer import L2Decay
from core.workspace import register
from models.layers import DeformableConvV2
from models.layers import LiteConv
import numpy as np


@register
class HMHead(nn.Module):
    """
    Args:
        ch_in (int): The channel number of input Tensor.
        ch_out (int): The channel number of output Tensor.
        num_classes (int): Number of classes.
        conv_num (int): The convolution number of hm_feat.
        dcn_head(bool): whether use dcn in head. False by default. 
        lite_head(bool): whether use lite version. False by default.
        norm_type (string): norm type, 'sync_bn', 'bn', 'gn' are optional.
            bn by default

    Return:
        Heatmap head output
    """
    __shared__ = ['num_classes', 'norm_type']

    def __init__(self, ch_in, ch_out=128, num_classes=80, conv_num=2,
        dcn_head=False, lite_head=False, norm_type='bn'):
        super(HMHead, self).__init__()
        _head_conv = []
        for i in range(conv_num):
            name = 'conv.{}'.format(i)
            if lite_head:
                lite_name = 'hm.' + name
                _head_conv.append(LiteConv(in_channels=ch_in if i == 0 else
                    ch_out, out_channels=ch_out, norm_type=norm_type))
            else:
                if dcn_head:
                    _head_conv.append(DeformableConvV2(in_channels=ch_in if
                        i == 0 else ch_out, out_channels=ch_out,
                        kernel_size=3, weight_attr=xavier_uniform()))
                else:
                    _head_conv.append(nn.GroupConv2d(in_channels=ch_in if i ==\
                        0 else ch_out, out_channels=ch_out, kernel_size=3,
                        padding=1, W_init=xavier_uniform(), b_init=\
                        xavier_uniform(), data_format='channels_first'))
                _head_conv.append(nn.ReLU())
        self.feat = nn.Sequential([*_head_conv])
        bias_init = float(-np.log((1 - 0.01) / 0.01))
        weight_attr = None if lite_head else xavier_uniform()
        self.head = nn.GroupConv2d(in_channels=ch_out, out_channels=\
            num_classes, kernel_size=1, W_init=weight_attr, b_init=\
            xavier_uniform(), padding=0, data_format='channels_first')

    def forward(self, feat):
        out = self.feat(feat)
        out = self.head(out)
        return out


@register
class WHHead(nn.Module):
    """
    Args:
        ch_in (int): The channel number of input Tensor.
        ch_out (int): The channel number of output Tensor.
        conv_num (int): The convolution number of wh_feat.
        dcn_head(bool): whether use dcn in head. False by default.
        lite_head(bool): whether use lite version. False by default.
        norm_type (string): norm type, 'sync_bn', 'bn', 'gn' are optional.
            bn by default
    Return:
        Width & Height head output
    """
    __shared__ = ['norm_type']

    def __init__(self, ch_in, ch_out=64, conv_num=2, dcn_head=False,
        lite_head=False, norm_type='bn'):
        super(WHHead, self).__init__()
        _head_conv = []
        for i in range(conv_num):
            name = 'conv.{}'.format(i)
            if lite_head:
                lite_name = 'wh.' + name
                _head_conv.append(LiteConv(in_channels=ch_in if i == 0 else
                    ch_out, out_channels=ch_out, norm_type=norm_type))
            else:
                if dcn_head:
                    _head_conv.append(DeformableConvV2(in_channels=ch_in if
                        i == 0 else ch_out, out_channels=ch_out,
                        kernel_size=3, weight_attr=xavier_uniform()))
                else:
                    _head_conv.append(nn.GroupConv2d(in_channels=ch_in if i ==\
                        0 else ch_out, out_channels=ch_out, kernel_size=3,
                        padding=1, W_init=xavier_uniform(), b_init=\
                        xavier_uniform(), data_format='channels_first'))
                _head_conv.append(nn.ReLU())
        weight_attr = None if lite_head else xavier_uniform()
        self.feat = nn.Sequential([*_head_conv])
        self.head = nn.GroupConv2d(in_channels=ch_out, out_channels=4,
            kernel_size=1, W_init=weight_attr, b_init=xavier_uniform(),
            padding=0, data_format='channels_first')

    def forward(self, feat):
        out = self.feat(feat)
        out = self.head(out)
        out = tensorlayerx.ops.relu(out)
        return out


@register
class TTFHead(nn.Module):
    """
    TTFHead
    Args:
        in_channels (int): the channel number of input to TTFHead.
        num_classes (int): the number of classes, 80 by default.
        hm_head_planes (int): the channel number in heatmap head,
            128 by default.
        wh_head_planes (int): the channel number in width & height head,
            64 by default.
        hm_head_conv_num (int): the number of convolution in heatmap head,
            2 by default.
        wh_head_conv_num (int): the number of convolution in width & height
            head, 2 by default.
        hm_loss (object): Instance of 'CTFocalLoss'.
        wh_loss (object): Instance of 'GIoULoss'.
        wh_offset_base (float): the base offset of width and height,
            16.0 by default.
        down_ratio (int): the actual down_ratio is calculated by base_down_ratio
            (default 16) and the number of upsample layers.
        lite_head(bool): whether use lite version. False by default.
        norm_type (string): norm type, 'sync_bn', 'bn', 'gn' are optional.
            bn by default
        ags_module(bool): whether use AGS module to reweight location feature.
            false by default.

    """
    __shared__ = ['num_classes', 'down_ratio', 'norm_type']
    __inject__ = ['hm_loss', 'wh_loss']

    def __init__(self, in_channels, num_classes=80, hm_head_planes=128,
        wh_head_planes=64, hm_head_conv_num=2, wh_head_conv_num=2, hm_loss=\
        'CTFocalLoss', wh_loss='GIoULoss', wh_offset_base=16.0, down_ratio=\
        4, dcn_head=False, lite_head=False, norm_type='bn', ags_module=False):
        super(TTFHead, self).__init__()
        self.in_channels = in_channels
        self.hm_head = HMHead(in_channels, hm_head_planes, num_classes,
            hm_head_conv_num, dcn_head, lite_head, norm_type)
        self.wh_head = WHHead(in_channels, wh_head_planes, wh_head_conv_num,
            dcn_head, lite_head, norm_type)
        self.hm_loss = hm_loss
        self.wh_loss = wh_loss
        self.wh_offset_base = wh_offset_base
        self.down_ratio = down_ratio
        self.ags_module = ags_module

    @classmethod
    def from_config(cls, cfg, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channels': input_shape.channels}

    def forward(self, feats):
        hm = self.hm_head(feats)
        wh = self.wh_head(feats) * self.wh_offset_base
        return hm, wh

    def filter_box_by_weight(self, pred, target, weight):
        """
        Filter out boxes where ttf_reg_weight is 0, only keep positive samples.
        """
        index = paddle2tlx.pd2tlx.ops.tlxops.tlx_nonzero(weight > 0)
        index.stop_gradient = True
        weight = tensorlayerx.gather_nd(weight, index)
        pred = tensorlayerx.gather_nd(pred, index)
        target = tensorlayerx.gather_nd(target, index)
        return pred, target, weight

    def filter_loc_by_weight(self, score, weight):
        index = paddle2tlx.pd2tlx.ops.tlxops.tlx_nonzero(weight > 0)
        index.stop_gradient = True
        score = tensorlayerx.gather_nd(score, index)
        return score

    def get_loss(self, pred_hm, pred_wh, target_hm, box_target, target_weight):
        pred_hm = tensorlayerx.ops.clip_by_value(tensorlayerx.ops.sigmoid(
            pred_hm), clip_value_min=0.0001, clip_value_max=1 - 0.0001)
        hm_loss = self.hm_loss(pred_hm, target_hm)
        H, W = target_hm.shape[2:]
        mask = tensorlayerx.reshape(target_weight, [-1, H, W])
        aa = tensorlayerx.reduce_sum(mask)
        avg_factor = aa + 0.0001
        base_step = self.down_ratio
        shifts_x = tensorlayerx.ops.arange(0, W * base_step, base_step,
            dtype='int32')
        shifts_y = tensorlayerx.ops.arange(0, H * base_step, base_step,
            dtype='int32')
        shift_y, shift_x = tensorlayerx.meshgrid([shifts_y, shifts_x])
        base_loc = tensorlayerx.ops.stack([shift_x, shift_y], axis=0)
        base_loc.stop_gradient = True
        pred_boxes = tensorlayerx.concat([0 - pred_wh[:, 0:2, :, :] +
            base_loc, pred_wh[:, 2:4] + base_loc], axis=1)
        pred_boxes = tensorlayerx.transpose(pred_boxes, [0, 2, 3, 1])
        boxes = tensorlayerx.transpose(box_target, [0, 2, 3, 1])
        boxes.stop_gradient = True
        if self.ags_module:
            pred_hm_max = tensorlayerx.reduce_max(pred_hm, axis=1, keepdims
                =True)
            pred_hm_max_softmax = tensorlayerx.ops.softmax(pred_hm_max, axis=1)
            pred_hm_max_softmax = tensorlayerx.transpose(pred_hm_max_softmax,
                [0, 2, 3, 1])
            pred_hm_max_softmax = self.filter_loc_by_weight(pred_hm_max_softmax
                , mask)
        else:
            pred_hm_max_softmax = None
        pred_boxes, boxes, mask = self.filter_box_by_weight(pred_boxes,
            boxes, mask)
        mask.stop_gradient = True
        wh_loss = self.wh_loss(pred_boxes, boxes, iou_weight=mask.unsqueeze
            (1), loc_reweight=pred_hm_max_softmax)
        wh_loss = wh_loss / avg_factor
        ttf_loss = {'hm_loss': hm_loss, 'wh_loss': wh_loss}
        return ttf_loss
