#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
import six
import numpy as np
from numbers import Integral

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle import to_tensor
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant, XavierUniform
from paddle.regularizer import L2Decay

from core.workspace import register, serializable
from models.bbox_utils import delta2bbox
from . import ops

from paddle.vision.ops import DeformConv2D

def _to_list(l):
    if isinstance(l, (list, tuple)):
        return list(l)
    return [l]


class DeformableConvV2(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 lr_scale=1,
                 regularizer=None,
                 skip_quant=False,
                 dcn_bias_regularizer=None,
                 dcn_bias_lr_scale=2.):
        super(DeformableConvV2, self).__init__()
        self.offset_channel = 2 * kernel_size**2
        self.mask_channel = kernel_size**2

        if lr_scale == 1 and regularizer is None:
            offset_bias_attr = ParamAttr(initializer=Constant(0.))
        else:
            offset_bias_attr = ParamAttr(
                initializer=Constant(0.),
                learning_rate=lr_scale,
                regularizer=regularizer)
        self.conv_offset = nn.Conv2D(
            in_channels,
            3 * kernel_size**2,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            weight_attr=ParamAttr(initializer=Constant(0.0)),
            bias_attr=offset_bias_attr)
        if skip_quant:
            self.conv_offset.skip_quant = True

        if bias_attr:
            self.conv_dcn = DeformConv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2 * dilation,
                dilation=dilation,
                groups=groups,
                weight_attr=ParamAttr(initializer=Constant(0.0)),
                bias_attr=ParamAttr(
                    initializer=Constant(value=0),
                    learning_rate=dcn_bias_lr_scale))
        else:
            self.conv_dcn = DeformConv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2 * dilation,
                dilation=dilation,
                groups=groups,
                weight_attr=ParamAttr(initializer=Constant(0.0)),
                bias_attr=False)


    def forward(self, x):
        offset_mask = self.conv_offset(x)
        offset, mask = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1)
        mask = F.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y


class ConvNormLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 norm_type='bn',
                 norm_decay=0.,
                 norm_groups=32,
                 use_dcn=False,
                 bias_on=False,
                 lr_scale=1.,
                 freeze_norm=False,
                 initializer=Normal(0., 0.01),
                 skip_quant=False,
                 dcn_lr_scale=2.,
                 dcn_regularizer=L2Decay(0.)):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn', None]

        if bias_on:
            bias_attr = ParamAttr(
                initializer=Constant(value=0.), learning_rate=lr_scale)
        else:
            bias_attr = False

        if not use_dcn:
            self.conv = nn.Conv2D(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                weight_attr=ParamAttr(
                    initializer=initializer, learning_rate=1.),
                bias_attr=bias_attr)
            # if skip_quant:
            #     self.conv.skip_quant = True
        else:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            self.conv = DeformableConvV2(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                weight_attr=ParamAttr(
                    initializer=initializer, learning_rate=1.),
                bias_attr=True,
                lr_scale=dcn_lr_scale,
                regularizer=dcn_regularizer,
                dcn_bias_regularizer=dcn_regularizer,
                dcn_bias_lr_scale=dcn_lr_scale,
                skip_quant=skip_quant)

        norm_lr = 0. if freeze_norm else 1.
        param_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        bias_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        if norm_type in ['bn', 'sync_bn']:
            self.norm = nn.BatchNorm2D(
                ch_out, weight_attr=param_attr, bias_attr=bias_attr)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(
                num_groups=norm_groups,
                num_channels=ch_out,)
        else:
            self.norm = None

    def forward(self, inputs):
        out = self.conv(inputs)
        if self.norm is not None:
            out = self.norm(out)
        return out


class LiteConv(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 with_act=True,
                 norm_type='sync_bn',
                 name=None):
        super(LiteConv, self).__init__()
        self.lite_conv = nn.Sequential()
        conv1 = ConvNormLayer(
            in_channels,
            in_channels,
            filter_size=5,
            stride=stride,
            groups=in_channels,
            norm_type=norm_type,
            initializer=XavierUniform())
        conv2 = ConvNormLayer(
            in_channels,
            out_channels,
            filter_size=1,
            stride=stride,
            norm_type=norm_type,
            initializer=XavierUniform())
        conv3 = ConvNormLayer(
            out_channels,
            out_channels,
            filter_size=1,
            stride=stride,
            norm_type=norm_type,
            initializer=XavierUniform())
        conv4 = ConvNormLayer(
            out_channels,
            out_channels,
            filter_size=5,
            stride=stride,
            groups=out_channels,
            norm_type=norm_type,
            initializer=XavierUniform())
        conv_list = [conv1, conv2, conv3, conv4]
        self.lite_conv.add_sublayer('conv1', conv1)
        self.lite_conv.add_sublayer('relu6_1', nn.ReLU6())
        self.lite_conv.add_sublayer('conv2', conv2)
        if with_act:
            self.lite_conv.add_sublayer('relu6_2', nn.ReLU6())
        self.lite_conv.add_sublayer('conv3', conv3)
        self.lite_conv.add_sublayer('relu6_3', nn.ReLU6())
        self.lite_conv.add_sublayer('conv4', conv4)
        if with_act:
            self.lite_conv.add_sublayer('relu6_4', nn.ReLU6())

    def forward(self, inputs):
        out = self.lite_conv(inputs)
        return out


class DropBlock(nn.Layer):
    def __init__(self, block_size, keep_prob, name=None, data_format='NCHW'):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890

        Args:
            block_size (int): block size
            keep_prob (int): keep probability
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = (1. - self.keep_prob) / (self.block_size**2)
            if self.data_format == 'NCHW' or self.data_format == 'channels_first':  # todo
                shape = x.shape[2:]
            else:
                shape = x.shape[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            matrix = paddle.cast(paddle.rand(x.shape) < gamma, x.dtype)
            mask_inv = F.max_pool2d(
                matrix,
                self.block_size,
                stride=1,
                padding=self.block_size // 2,
                data_format=self.data_format)
            mask = 1. - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y


@register
@serializable
class AnchorGeneratorSSD(object):
    def __init__(self,
                 steps=[8, 16, 32, 64, 100, 300],
                 aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
                 min_ratio=15,
                 max_ratio=90,
                 base_size=300,
                 min_sizes=[30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
                 max_sizes=[60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
                 offset=0.5,
                 flip=True,
                 clip=False,
                 min_max_aspect_ratios_order=False):
        self.steps = steps
        self.aspect_ratios = aspect_ratios
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.base_size = base_size
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.offset = offset
        self.flip = flip
        self.clip = clip
        self.min_max_aspect_ratios_order = min_max_aspect_ratios_order

        if self.min_sizes == [] and self.max_sizes == []:
            num_layer = len(aspect_ratios)
            step = int(
                math.floor(((self.max_ratio - self.min_ratio)) / (num_layer - 2
                                                                  )))
            for ratio in six.moves.range(self.min_ratio, self.max_ratio + 1,
                                         step):
                self.min_sizes.append(self.base_size * ratio / 100.)
                self.max_sizes.append(self.base_size * (ratio + step) / 100.)
            self.min_sizes = [self.base_size * .10] + self.min_sizes
            self.max_sizes = [self.base_size * .20] + self.max_sizes

        self.num_priors = []
        for aspect_ratio, min_size, max_size in zip(
                aspect_ratios, self.min_sizes, self.max_sizes):
            if isinstance(min_size, (list, tuple)):
                self.num_priors.append(
                    len(_to_list(min_size)) + len(_to_list(max_size)))
            else:
                self.num_priors.append((len(aspect_ratio) * 2 + 1) * len(
                    _to_list(min_size)) + len(_to_list(max_size)))

    def __call__(self, inputs, image):
        boxes = []
        for input, min_size, max_size, aspect_ratio, step in zip(
                inputs, self.min_sizes, self.max_sizes, self.aspect_ratios,
                self.steps):
            box, _ = ops.prior_box(
                input=input,
                image=image,
                min_sizes=_to_list(min_size),
                max_sizes=_to_list(max_size),
                aspect_ratios=aspect_ratio,
                flip=self.flip,
                clip=self.clip,
                steps=[step, step],
                offset=self.offset,
                min_max_aspect_ratios_order=self.min_max_aspect_ratios_order)
            boxes.append(paddle.reshape(box, [-1, 4]))
        return boxes


@register
@serializable
class RCNNBox(object):
    __shared__ = ['num_classes', 'export_onnx']

    def __init__(self,
                 prior_box_var=[10., 10., 5., 5.],
                 code_type="decode_center_size",
                 box_normalized=False,
                 num_classes=80,
                 export_onnx=False):
        super(RCNNBox, self).__init__()
        self.prior_box_var = prior_box_var
        self.code_type = code_type
        self.box_normalized = box_normalized
        self.num_classes = num_classes
        self.export_onnx = export_onnx

    def __call__(self, bbox_head_out, rois, im_shape, scale_factor):
        bbox_pred = bbox_head_out[0]
        cls_prob = bbox_head_out[1]
        roi = rois[0]
        rois_num = rois[1]

        if self.export_onnx:
            onnx_rois_num_per_im = rois_num[0]
            origin_shape = paddle.expand(im_shape[0, :],
                                         [onnx_rois_num_per_im, 2])

        else:
            origin_shape_list = []
            if isinstance(roi, list):
                batch_size = len(roi)
            else:
                batch_size = paddle.slice(paddle.shape(im_shape), [0], [0], [1])

            # bbox_pred.shape: [N, C*4]
            for idx in range(batch_size):
                rois_num_per_im = rois_num[idx]
                expand_im_shape = paddle.expand(im_shape[idx, :],
                                                [rois_num_per_im, 2])
                origin_shape_list.append(expand_im_shape)

            origin_shape = paddle.concat(origin_shape_list)

        # bbox_pred.shape: [N, C*4]
        # C=num_classes in faster/mask rcnn(bbox_head), C=1 in cascade rcnn(cascade_head)
        bbox = paddle.concat(roi)
        bbox = delta2bbox(bbox_pred, bbox, self.prior_box_var)
        scores = cls_prob[:, :-1]

        # bbox.shape: [N, C, 4]
        # bbox.shape[1] must be equal to scores.shape[1]
        total_num = bbox.shape[0]
        bbox_dim = bbox.shape[-1]
        bbox = paddle.expand(bbox, [total_num, self.num_classes, bbox_dim])

        origin_h = paddle.unsqueeze(origin_shape[:, 0], axis=1)
        origin_w = paddle.unsqueeze(origin_shape[:, 1], axis=1)
        zeros = paddle.zeros_like(origin_h)
        x1 = paddle.maximum(paddle.minimum(bbox[:, :, 0], origin_w), zeros)
        y1 = paddle.maximum(paddle.minimum(bbox[:, :, 1], origin_h), zeros)
        x2 = paddle.maximum(paddle.minimum(bbox[:, :, 2], origin_w), zeros)
        y2 = paddle.maximum(paddle.minimum(bbox[:, :, 3], origin_h), zeros)
        bbox = paddle.stack([x1, y1, x2, y2], axis=-1)
        bboxes = (bbox, rois_num)
        return bboxes, scores


@register
@serializable
class MultiClassNMS(object):
    def __init__(self,
                 score_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 nms_threshold=.5,
                 normalized=True,
                 nms_eta=1.0,
                 return_index=False,
                 return_rois_num=True,
                 trt=False):
        super(MultiClassNMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.return_index = return_index
        self.return_rois_num = return_rois_num
        self.trt = trt

    def __call__(self, bboxes, score, background_label=-1):
        """
        bboxes (Tensor|List[Tensor]): 1. (Tensor) Predicted bboxes with shape 
                                         [N, M, 4], N is the batch size and M
                                         is the number of bboxes
                                      2. (List[Tensor]) bboxes and bbox_num,
                                         bboxes have shape of [M, C, 4], C
                                         is the class number and bbox_num means
                                         the number of bboxes of each batch with
                                         shape [N,] 
        score (Tensor): Predicted scores with shape [N, C, M] or [M, C]
        background_label (int): Ignore the background label; For example, RCNN
                                is num_classes and YOLO is -1. 
        """
        kwargs = self.__dict__.copy()
        if isinstance(bboxes, tuple):
            bboxes, bbox_num = bboxes
            kwargs.update({'rois_num': bbox_num})
        if background_label > -1:
            kwargs.update({'background_label': background_label})
        kwargs.pop('trt')

        return ops.multiclass_nms(bboxes, score, **kwargs)


@register
@serializable
class YOLOBox(object):
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 conf_thresh=0.005,
                 downsample_ratio=32,
                 clip_bbox=True,
                 scale_x_y=1.):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.downsample_ratio = downsample_ratio
        self.clip_bbox = clip_bbox
        self.scale_x_y = scale_x_y

    def __call__(self,
                 yolo_head_out,
                 anchors,
                 im_shape,
                 scale_factor,
                 var_weight=None):
        boxes_list = []
        scores_list = []
        origin_shape = im_shape / scale_factor
        origin_shape = paddle.cast(origin_shape, 'int32')
        for i, head_out in enumerate(yolo_head_out):
            boxes, scores = paddle.vision.ops.yolo_box(
                head_out,
                origin_shape,
                anchors[i],
                self.num_classes,
                self.conf_thresh,
                self.downsample_ratio // 2**i,
                self.clip_bbox,
                scale_x_y=self.scale_x_y)
            boxes_list.append(boxes)
            scores_list.append(paddle.transpose(scores, perm=[0, 2, 1]))
        yolo_boxes = paddle.concat(boxes_list, axis=1)
        yolo_scores = paddle.concat(scores_list, axis=2)
        return yolo_boxes, yolo_scores


@register
@serializable
class SSDBox(object):
    def __init__(self,
                 is_normalized=True,
                 prior_box_var=[0.1, 0.1, 0.2, 0.2],
                 use_fuse_decode=False):
        self.is_normalized = is_normalized
        self.norm_delta = float(not self.is_normalized)
        self.prior_box_var = prior_box_var
        self.use_fuse_decode = use_fuse_decode


    def __call__(self,
                 preds,
                 prior_boxes,
                 im_shape,
                 scale_factor,
                 var_weight=None):
        boxes, scores = preds
        boxes = paddle.concat(boxes, axis=1)
        prior_boxes = paddle.concat(prior_boxes)
        if self.use_fuse_decode:
            # output_boxes = ops.box_coder(
            #     prior_boxes,
            #     self.prior_box_var,
            #     boxes,
            #     code_type="decode_center_size",
            #     box_normalized=self.is_normalized)
            print(f"**********************SSDBox use_fuse_decode start***********************")
            print(f"SSDBox use_fuse_decode={self.is_normalizeduse_fuse_decode}")
            print(f"**********************SSDBox use_fuse_decode end***********************")
            raise "***self.is_normalizeduse_fuse_decode should not be True"
        else:
            pb_w = prior_boxes[:, 2] - prior_boxes[:, 0] + self.norm_delta
            pb_h = prior_boxes[:, 3] - prior_boxes[:, 1] + self.norm_delta
            pb_x = prior_boxes[:, 0] + pb_w * 0.5
            pb_y = prior_boxes[:, 1] + pb_h * 0.5
            out_x = pb_x + boxes[:, :, 0] * pb_w * self.prior_box_var[0]
            out_y = pb_y + boxes[:, :, 1] * pb_h * self.prior_box_var[1]
            aa = paddle.exp(boxes[:, :, 2] * self.prior_box_var[2])
            bb = paddle.exp(boxes[:, :, 3] * self.prior_box_var[3])
            out_w = aa * pb_w
            out_h = bb * pb_h
            output_boxes = paddle.stack(
                [
                    out_x - out_w / 2., out_y - out_h / 2., out_x + out_w / 2.,
                    out_y + out_h / 2.
                ],
                axis=-1)

        if self.is_normalized:
            h = (im_shape[:, 0] / scale_factor[:, 0]).unsqueeze(-1)
            w = (im_shape[:, 1] / scale_factor[:, 1]).unsqueeze(-1)
            im_shape = paddle.stack([w, h, w, h], axis=-1)
            output_boxes *= im_shape
        else:
            output_boxes[..., -2:] -= 1.0
        output_scores = F.softmax(paddle.concat(
            scores, axis=1)).transpose([0, 2, 1])

        return output_boxes, output_scores


@register
@serializable
class FCOSBox(object):
    __shared__ = ['num_classes']

    def __init__(self, num_classes=80):
        super(FCOSBox, self).__init__()
        self.num_classes = num_classes

    def _merge_hw(self, inputs, ch_type="channel_first"):
        """
        Merge h and w of the feature map into one dimension.
        Args:
            inputs (Tensor): Tensor of the input feature map
            ch_type (str): "channel_first" or "channel_last" style
        Return:
            new_shape (Tensor): The new shape after h and w merged
        """
        shape_ = paddle.shape(inputs)
        bs, ch, hi, wi = shape_[0], shape_[1], shape_[2], shape_[3]
        img_size = hi * wi
        img_size.stop_gradient = True
        if ch_type == "channel_first":
            new_shape = paddle.concat([bs, ch, img_size])
        elif ch_type == "channel_last":
            new_shape = paddle.concat([bs, img_size, ch])
        else:
            raise KeyError("Wrong ch_type %s" % ch_type)
        new_shape.stop_gradient = True
        return new_shape

    def _postprocessing_by_level(self, locations, box_cls, box_reg, box_ctn,
                                 scale_factor):
        """
        Postprocess each layer of the output with corresponding locations.
        Args:
            locations (Tensor): anchor points for current layer, [H*W, 2]
            box_cls (Tensor): categories prediction, [N, C, H, W], 
                C is the number of classes
            box_reg (Tensor): bounding box prediction, [N, 4, H, W]
            box_ctn (Tensor): centerness prediction, [N, 1, H, W]
            scale_factor (Tensor): [h_scale, w_scale] for input images
        Return:
            box_cls_ch_last (Tensor): score for each category, in [N, C, M]
                C is the number of classes and M is the number of anchor points
            box_reg_decoding (Tensor): decoded bounding box, in [N, M, 4]
                last dimension is [x1, y1, x2, y2]
        """
        act_shape_cls = self._merge_hw(box_cls)
        box_cls_ch_last = paddle.reshape(box_cls, shape=act_shape_cls)
        box_cls_ch_last = F.sigmoid(box_cls_ch_last)

        act_shape_reg = self._merge_hw(box_reg)
        box_reg_ch_last = paddle.reshape(box_reg, shape=act_shape_reg)
        box_reg_ch_last = paddle.transpose(box_reg_ch_last, perm=[0, 2, 1])
        box_reg_decoding = paddle.stack(
            [
                locations[:, 0] - box_reg_ch_last[:, :, 0],
                locations[:, 1] - box_reg_ch_last[:, :, 1],
                locations[:, 0] + box_reg_ch_last[:, :, 2],
                locations[:, 1] + box_reg_ch_last[:, :, 3]
            ],
            axis=1)
        box_reg_decoding = paddle.transpose(box_reg_decoding, perm=[0, 2, 1])

        act_shape_ctn = self._merge_hw(box_ctn)
        box_ctn_ch_last = paddle.reshape(box_ctn, shape=act_shape_ctn)
        box_ctn_ch_last = F.sigmoid(box_ctn_ch_last)

        # recover the location to original image
        im_scale = paddle.concat([scale_factor, scale_factor], axis=1)
        im_scale = paddle.expand(im_scale, [box_reg_decoding.shape[0], 4])
        im_scale = paddle.reshape(im_scale, [box_reg_decoding.shape[0], -1, 4])
        box_reg_decoding = box_reg_decoding / im_scale
        box_cls_ch_last = box_cls_ch_last * box_ctn_ch_last
        return box_cls_ch_last, box_reg_decoding

    def __call__(self, locations, cls_logits, bboxes_reg, centerness,
                 scale_factor):
        pred_boxes_ = []
        pred_scores_ = []
        for pts, cls, box, ctn in zip(locations, cls_logits, bboxes_reg,
                                      centerness):
            pred_scores_lvl, pred_boxes_lvl = self._postprocessing_by_level(
                pts, cls, box, ctn, scale_factor)
            pred_boxes_.append(pred_boxes_lvl)
            pred_scores_.append(pred_scores_lvl)
        pred_boxes = paddle.concat(pred_boxes_, axis=1)
        pred_scores = paddle.concat(pred_scores_, axis=2)
        return pred_boxes, pred_scores


@register
class TTFBox(object):
    __shared__ = ['down_ratio']

    def __init__(self, max_per_img=100, score_thresh=0.01, down_ratio=4):
        super(TTFBox, self).__init__()
        self.max_per_img = max_per_img
        self.score_thresh = score_thresh
        self.down_ratio = down_ratio

    def _simple_nms(self, heat, kernel=3):
        """
        Use maxpool to filter the max score, get local peaks.
        """
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = paddle.cast(hmax == heat, 'float32')
        return heat * keep

    def _topk(self, scores):
        """
        Select top k scores and decode to get xy coordinates.
        """
        k = self.max_per_img
        shape_fm = paddle.shape(scores)
        shape_fm.stop_gradient = True
        cat, height, width = shape_fm[1], shape_fm[2], shape_fm[3]
        # batch size is 1
        scores_r = paddle.reshape(scores, [cat, -1])
        topk_scores, topk_inds = paddle.topk(scores_r, k)
        topk_ys = topk_inds // width
        topk_xs = topk_inds % width

        topk_score_r = paddle.reshape(topk_scores, [-1])
        topk_score, topk_ind = paddle.topk(topk_score_r, k)
        k_t = paddle.full(shape=paddle.shape(topk_ind), fill_value=k, dtype='int64')
        topk_clses = paddle.cast(paddle.floor_divide(topk_ind, k_t), 'float32')

        topk_inds = paddle.reshape(topk_inds, [-1])
        topk_ys = paddle.reshape(topk_ys, [-1, 1])
        topk_xs = paddle.reshape(topk_xs, [-1, 1])
        topk_inds = paddle.gather(topk_inds, topk_ind)
        topk_ys = paddle.gather(topk_ys, topk_ind)
        topk_xs = paddle.gather(topk_xs, topk_ind)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _decode(self, hm, wh, im_shape, scale_factor):
        heatmap = F.sigmoid(hm)
        heat = self._simple_nms(heatmap)
        scores, inds, clses, ys, xs = self._topk(heat)
        aa = paddle.cast(ys, 'float32')
        bb = paddle.cast(xs, 'float32')
        ys = aa * self.down_ratio
        xs = bb * self.down_ratio
        scores = paddle.tensor.unsqueeze(scores, [1])
        clses = paddle.tensor.unsqueeze(clses, [1])

        wh_t = paddle.transpose(wh, [0, 2, 3, 1])
        wh = paddle.reshape(wh_t, [-1, paddle.shape(wh_t)[-1]])
        wh = paddle.gather(wh, inds)

        x1 = xs - wh[:, 0:1]
        y1 = ys - wh[:, 1:2]
        x2 = xs + wh[:, 2:3]
        y2 = ys + wh[:, 3:4]

        bboxes = paddle.concat([x1, y1, x2, y2], axis=1)

        scale_y = scale_factor[:, 0:1]
        scale_x = scale_factor[:, 1:2]
        scale_expand = paddle.concat(
            [scale_x, scale_y, scale_x, scale_y], axis=1)
        boxes_shape = paddle.shape(bboxes)
        boxes_shape.stop_gradient = True
        scale_expand = paddle.expand(scale_expand, shape=boxes_shape)
        bboxes = paddle.divide(bboxes, scale_expand)
        results = paddle.concat([clses, scores, bboxes], axis=1)
        # hack: append result with cls=-1 and score=1. to avoid all scores
        # are less than score_thresh which may cause error in gather.
        fill_r = paddle.to_tensor(np.array([[-1, 1, 0, 0, 0, 0]]))
        fill_r = paddle.cast(fill_r, results.dtype)
        results = paddle.concat([results, fill_r])
        scores = results[:, 1]
        valid_ind = paddle.nonzero(scores > self.score_thresh)
        results = paddle.gather(results, valid_ind)
        return results, paddle.shape(results)[0:1]

    def __call__(self, hm, wh, im_shape, scale_factor):
        results = []
        results_num = []
        for i in range(scale_factor.shape[0]):
            result, num = self._decode(hm[i:i + 1, ], wh[i:i + 1, ],
                                       im_shape[i:i + 1, ],
                                       scale_factor[i:i + 1, ])
            results.append(result)
            results_num.append(num)
        results = paddle.concat(results, axis=0)
        results_num = paddle.concat(results_num, axis=0)
        return results, results_num


@register
@serializable
class MaskMatrixNMS(object):
    """
    Matrix NMS for multi-class masks.
    Args:
        update_threshold (float): Updated threshold of categroy score in second time.
        pre_nms_top_n (int): Number of total instance to be kept per image before NMS
        post_nms_top_n (int): Number of total instance to be kept per image after NMS.
        kernel (str):  'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
    Input:
        seg_preds (Variable): shape (n, h, w), segmentation feature maps
        seg_masks (Variable): shape (n, h, w), segmentation feature maps
        cate_labels (Variable): shape (n), mask labels in descending order
        cate_scores (Variable): shape (n), mask scores in descending order
        sum_masks (Variable): a float tensor of the sum of seg_masks
    Returns:
        Variable: cate_scores, tensors of shape (n)
    """

    def __init__(self,
                 update_threshold=0.05,
                 pre_nms_top_n=500,
                 post_nms_top_n=100,
                 kernel='gaussian',
                 sigma=2.0):
        super(MaskMatrixNMS, self).__init__()
        self.update_threshold = update_threshold
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.kernel = kernel
        self.sigma = sigma

    def _sort_score(self, scores, top_num):
        if paddle.shape(scores)[0] > top_num:
            return paddle.topk(scores, top_num)[1]
        else:
            return paddle.argsort(scores, descending=True)

    def __call__(self,
                 seg_preds,
                 seg_masks,
                 cate_labels,
                 cate_scores,
                 sum_masks=None):
        # sort and keep top nms_pre
        sort_inds = self._sort_score(cate_scores, self.pre_nms_top_n)
        seg_masks = paddle.gather(seg_masks, index=sort_inds)
        seg_preds = paddle.gather(seg_preds, index=sort_inds)
        sum_masks = paddle.gather(sum_masks, index=sort_inds)
        cate_scores = paddle.gather(cate_scores, index=sort_inds)
        cate_labels = paddle.gather(cate_labels, index=sort_inds)

        seg_masks = paddle.flatten(seg_masks, start_axis=1, stop_axis=-1)
        # inter.
        inter_matrix = paddle.mm(seg_masks, paddle.transpose(seg_masks, [1, 0]))
        n_samples = paddle.shape(cate_labels)
        # union.
        sum_masks_x = paddle.expand(sum_masks, shape=[n_samples, n_samples])
        # iou.
        aa = paddle.transpose(sum_masks_x, [1, 0])
        iou_matrix = (inter_matrix / (
            sum_masks_x + aa - inter_matrix))
        iou_matrix = paddle.triu(iou_matrix, diagonal=1)
        # label_specific matrix.
        cate_labels_x = paddle.expand(cate_labels, shape=[n_samples, n_samples])
        label_matrix = paddle.cast(
            (cate_labels_x == paddle.transpose(cate_labels_x, [1, 0])),
            'float32')
        label_matrix = paddle.triu(label_matrix, diagonal=1)

        # IoU compensation
        compensate_iou = paddle.max((iou_matrix * label_matrix), axis=0)
        compensate_iou = paddle.expand(
            compensate_iou, shape=[n_samples, n_samples])
        compensate_iou = paddle.transpose(compensate_iou, [1, 0])

        # IoU decay
        decay_iou = iou_matrix * label_matrix

        # matrix nms
        if self.kernel == 'gaussian':
            decay_matrix = paddle.exp(-1 * self.sigma * (decay_iou**2))
            compensate_matrix = paddle.exp(-1 * self.sigma *
                                           (compensate_iou**2))
            decay_coefficient = paddle.min(decay_matrix / compensate_matrix,
                                           axis=0)
        elif self.kernel == 'linear':
            decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
            decay_coefficient = paddle.min(decay_matrix, axis=0)
        else:
            raise NotImplementedError

        # update the score.
        cate_scores = cate_scores * decay_coefficient
        y = paddle.zeros(shape=paddle.shape(cate_scores), dtype='float32')
        keep = paddle.where(cate_scores >= self.update_threshold, cate_scores,
                            y)
        keep = paddle.nonzero(keep)
        keep = paddle.squeeze(keep, axis=[1])
        # Prevent empty and increase fake data
        aa = paddle.shape(cate_scores)
        keep = paddle.concat(
            [keep, paddle.cast(aa[0] - 1, 'int64')])

        seg_preds = paddle.gather(seg_preds, index=keep)
        cate_scores = paddle.gather(cate_scores, index=keep)
        cate_labels = paddle.gather(cate_labels, index=keep)

        # sort and keep top_k
        sort_inds = self._sort_score(cate_scores, self.post_nms_top_n)
        seg_preds = paddle.gather(seg_preds, index=sort_inds)
        cate_scores = paddle.gather(cate_scores, index=sort_inds)
        cate_labels = paddle.gather(cate_labels, index=sort_inds)
        return seg_preds, cate_scores, cate_labels
