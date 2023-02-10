import tensorlayerx as tlx
import paddle
import paddle2tlx
import numpy as np
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import random_normal
from tensorlayerx.nn.initializers import XavierUniform
from tensorlayerx.nn.initializers import HeNormal
from paddle.regularizer import L2Decay
from core.workspace import register
from core.workspace import create
from .roi_extractor import RoIAlign
from ..shape_spec import ShapeSpec
from ..bbox_utils import bbox2delta
from ..cls_utils import _get_class_default_kwargs
from models.layers import ConvNormLayer
__all__ = ['TwoFCHead', 'XConvNormHead', 'BBoxHead']


@register
class TwoFCHead(nn.Module):
    """
    RCNN bbox head with Two fc layers to extract feature

    Args:
        in_channel (int): Input channel which can be derived by from_config
        out_channel (int): Output channel
        resolution (int): Resolution of input feature map, default 7
    """

    def __init__(self, in_channel=256, out_channel=1024, resolution=7):
        super(TwoFCHead, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        fan = in_channel * resolution * resolution
        self.fc6 = nn.Linear(in_features=in_channel * resolution *
            resolution, out_features=out_channel)
        self.fc6.skip_quant = True
        self.fc7 = nn.Linear(in_features=out_channel, out_features=out_channel)
        self.fc7.skip_quant = True

    @classmethod
    def from_config(cls, cfg, input_shape):
        s = input_shape
        s = s[0] if isinstance(s, (list, tuple)) else s
        return {'in_channel': s.channels}

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.out_channel)]

    def forward(self, rois_feat):
        rois_feat = tensorlayerx.flatten(rois_feat, start_axis=1, stop_axis=-1)
        fc6 = self.fc6(rois_feat)
        fc6 = tensorlayerx.ops.relu(fc6)
        fc7 = self.fc7(fc6)
        fc7 = tensorlayerx.ops.relu(fc7)
        return fc7


@register
class XConvNormHead(nn.Module):
    __shared__ = ['norm_type', 'freeze_norm']
    """
    RCNN bbox head with serveral convolution layers

    Args:
        in_channel (int): Input channels which can be derived by from_config
        num_convs (int): The number of conv layers
        conv_dim (int): The number of channels for the conv layers
        out_channel (int): Output channels
        resolution (int): Resolution of input feature map
        norm_type (string): Norm type, bn, gn, sync_bn are available, 
            default `gn`
        freeze_norm (bool): Whether to freeze the norm
        stage_name (string): Prefix name for conv layer,  '' by default
    """

    def __init__(self, in_channel=256, num_convs=4, conv_dim=256,
        out_channel=1024, resolution=7, norm_type='gn', freeze_norm=False,
        stage_name=''):
        super(XConvNormHead, self).__init__()
        self.in_channel = in_channel
        self.num_convs = num_convs
        self.conv_dim = conv_dim
        self.out_channel = out_channel
        self.norm_type = norm_type
        self.freeze_norm = freeze_norm
        self.bbox_head_convs = []
        fan = conv_dim * 3 * 3
        initializer = HeNormal(fan_in=fan)
        for i in range(self.num_convs):
            in_c = in_channel if i == 0 else conv_dim
            head_conv_name = stage_name + 'bbox_head_conv{}'.format(i)
            head_conv = self.add_sublayer(head_conv_name, ConvNormLayer(
                ch_in=in_c, ch_out=conv_dim, filter_size=3, stride=1,
                norm_type=self.norm_type, freeze_norm=self.freeze_norm,
                initializer=initializer))
            self.bbox_head_convs.append(head_conv)
        fan = conv_dim * resolution * resolution
        self.fc6 = nn.Linear(in_features=conv_dim * resolution * resolution,
            out_features=out_channel, b_init=tensorlayerx.nn.initializers.
            xavier_uniform())

    @classmethod
    def from_config(cls, cfg, input_shape):
        s = input_shape
        s = s[0] if isinstance(s, (list, tuple)) else s
        return {'in_channel': s.channels}

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.out_channel)]

    def forward(self, rois_feat):
        for i in range(self.num_convs):
            rois_feat = tensorlayerx.ops.relu(self.bbox_head_convs[i](
                rois_feat))
        rois_feat = tensorlayerx.flatten(rois_feat, start_axis=1, stop_axis=-1)
        fc6 = tensorlayerx.ops.relu(self.fc6(rois_feat))
        return fc6


@register
class BBoxHead(nn.Module):
    __shared__ = ['num_classes']
    __inject__ = ['bbox_assigner', 'bbox_loss']
    """
    RCNN bbox head

    Args:
        head (nn.Layer): Extract feature in bbox head
        in_channel (int): Input channel after RoI extractor
        roi_extractor (object): The module of RoI Extractor
        bbox_assigner (object): The module of Box Assigner, label and sample the 
            box.
        with_pool (bool): Whether to use pooling for the RoI feature.
        num_classes (int): The number of classes
        bbox_weight (List[float]): The weight to get the decode box 
    """

    def __init__(self, head, in_channel, roi_extractor=\
        _get_class_default_kwargs(RoIAlign), bbox_assigner='BboxAssigner',
        with_pool=False, num_classes=80, bbox_weight=[10.0, 10.0, 5.0, 5.0],
        bbox_loss=None, loss_normalize_pos=False):
        super(BBoxHead, self).__init__()
        self.head = head
        self.roi_extractor = roi_extractor
        if isinstance(roi_extractor, dict):
            self.roi_extractor = RoIAlign(**roi_extractor)
        self.bbox_assigner = bbox_assigner
        self.with_pool = with_pool
        self.num_classes = num_classes
        self.bbox_weight = bbox_weight
        self.bbox_loss = bbox_loss
        self.loss_normalize_pos = loss_normalize_pos
        self.bbox_score = nn.Linear(in_features=in_channel, out_features=\
            self.num_classes + 1)
        self.bbox_score.skip_quant = True
        self.bbox_delta = nn.Linear(in_features=in_channel, out_features=4 *
            self.num_classes)
        self.bbox_delta.skip_quant = True
        self.assigned_label = None
        self.assigned_rois = None

    @classmethod
    def from_config(cls, cfg, input_shape):
        roi_pooler = cfg['roi_extractor']
        assert isinstance(roi_pooler, dict)
        kwargs = RoIAlign.from_config(cfg, input_shape)
        roi_pooler.update(kwargs)
        kwargs = {'input_shape': input_shape}
        head = create(cfg['head'], **kwargs)
        return {'roi_extractor': roi_pooler, 'head': head, 'in_channel':
            head.out_shape[0].channels}

    def forward(self, body_feats=None, rois=None, rois_num=None, inputs=None):
        """
        body_feats (list[Tensor]): Feature maps from backbone
        rois (list[Tensor]): RoIs generated from RPN module
        rois_num (Tensor): The number of RoIs in each image
        inputs (dict{Tensor}): The ground-truth of image
        """
        if self.training:
            rois, rois_num, targets = self.bbox_assigner(rois, rois_num, inputs
                )
            self.assigned_rois = rois, rois_num
            self.assigned_targets = targets
        rois_feat = self.roi_extractor(body_feats, rois, rois_num)
        bbox_feat = self.head(rois_feat)
        if self.with_pool:
            feat = paddle.nn.functional.adaptive_avg_pool2d(bbox_feat,
                output_size=1)
            feat = tensorlayerx.ops.squeeze(feat, axis=[2, 3])
        else:
            feat = bbox_feat
        scores = self.bbox_score(feat)
        deltas = self.bbox_delta(feat)
        if self.training:
            loss = self.get_loss(scores, deltas, targets, rois, self.
                bbox_weight, loss_normalize_pos=self.loss_normalize_pos)
            return loss, bbox_feat
        else:
            pred = self.get_prediction(scores, deltas)
            return pred, self.head

    def get_loss(self, scores, deltas, targets, rois, bbox_weight,
        loss_normalize_pos=False):
        """
        scores (Tensor): scores from bbox head outputs
        deltas (Tensor): deltas from bbox head outputs
        targets (list[List[Tensor]]): bbox targets containing tgt_labels, tgt_bboxes and tgt_gt_inds
        rois (List[Tensor]): RoIs generated in each batch
        """
        cls_name = 'loss_bbox_cls'
        reg_name = 'loss_bbox_reg'
        loss_bbox = {}
        tgt_labels, tgt_bboxes, tgt_gt_inds = targets
        tgt_labels = tensorlayerx.concat(tgt_labels) if len(tgt_labels
            ) > 1 else tgt_labels[0]
        valid_inds = paddle2tlx.pd2tlx.ops.tlxops.tlx_nonzero(tgt_labels >= 0
            ).flatten()
        if valid_inds.shape[0] == 0:
            loss_bbox[cls_name] = tensorlayerx.zeros([1], dtype='float32')
        else:
            tgt_labels = tgt_labels.cast('int64')
            tgt_labels.stop_gradient = True
            if not loss_normalize_pos:
                loss_bbox_cls = paddle2tlx.pd2tlx.ops.tlxops.tlx_cross_entropy(
                    input=scores, label=tgt_labels, reduction='mean')
            else:
                loss_bbox_cls = F.cross_entropy(input=scores, label=\
                    tgt_labels, reduction='none').sum() / (tgt_labels.shape
                    [0] + 1e-07)
            loss_bbox[cls_name] = loss_bbox_cls
        cls_agnostic_bbox_reg = deltas.shape[1] == 4
        fg_inds = paddle2tlx.pd2tlx.ops.tlxops.tlx_nonzero(tensorlayerx.ops
            .logical_and(tgt_labels >= 0, tgt_labels < self.num_classes)
            ).flatten()
        if fg_inds.numel() == 0:
            loss_bbox[reg_name] = tensorlayerx.zeros([1], dtype='float32')
            return loss_bbox
        if cls_agnostic_bbox_reg:
            reg_delta = tensorlayerx.gather(deltas, fg_inds)
        else:
            fg_gt_classes = tensorlayerx.gather(tgt_labels, fg_inds)
            reg_row_inds = tensorlayerx.ops.arange(fg_gt_classes.shape[0]
                ).unsqueeze(1)
            reg_row_inds = tensorlayerx.tile(reg_row_inds, [1, 4]).reshape([
                -1, 1])
            aa = tensorlayerx.ops.arange(4)
            reg_col_inds = 4 * fg_gt_classes.unsqueeze(1) + aa
            reg_col_inds = reg_col_inds.reshape([-1, 1])
            reg_inds = tensorlayerx.concat([reg_row_inds, reg_col_inds], axis=1
                )
            reg_delta = tensorlayerx.gather(deltas, fg_inds)
            reg_delta = tensorlayerx.gather_nd(reg_delta, reg_inds).reshape([
                -1, 4])
        rois = tensorlayerx.concat(rois) if len(rois) > 1 else rois[0]
        tgt_bboxes = tensorlayerx.concat(tgt_bboxes) if len(tgt_bboxes
            ) > 1 else tgt_bboxes[0]
        reg_target = bbox2delta(rois, tgt_bboxes, bbox_weight)
        reg_target = tensorlayerx.gather(reg_target, fg_inds)
        reg_target.stop_gradient = True
        if self.bbox_loss is not None:
            reg_delta = self.bbox_transform(reg_delta)
            reg_target = self.bbox_transform(reg_target)
            if not loss_normalize_pos:
                loss_bbox_reg = self.bbox_loss(reg_delta, reg_target).sum(
                    ) / tgt_labels.shape[0]
                loss_bbox_reg *= self.num_classes
            else:
                loss_bbox_reg = self.bbox_loss(reg_delta, reg_target).sum() / (
                    tgt_labels.shape[0] + 1e-07)
        else:
            aa = tensorlayerx.ops.abs(reg_delta - reg_target)
            loss_bbox_reg = aa.sum() / tgt_labels.shape[0]
        loss_bbox[reg_name] = loss_bbox_reg
        return loss_bbox

    def bbox_transform(self, deltas, weights=[0.1, 0.1, 0.2, 0.2]):
        wx, wy, ww, wh = weights
        deltas = tensorlayerx.reshape(deltas, shape=(0, -1, 4))
        aa = paddle.slice(deltas, axes=[2], starts=[0], ends=[1])
        bb = paddle.slice(deltas, axes=[2], starts=[1], ends=[2])
        cc = paddle.slice(deltas, axes=[2], starts=[2], ends=[3])
        dd = paddle.slice(deltas, axes=[2], starts=[3], ends=[4])
        dx = aa * wx
        dy = bb * wy
        dw = cc * ww
        dh = dd * wh
        dw = tensorlayerx.ops.clip_by_value(dw, -10000000000.0, np.log(
            1000.0 / 16), clip_value_min=None, clip_value_max=None)
        dh = tensorlayerx.ops.clip_by_value(dh, -10000000000.0, np.log(
            1000.0 / 16), clip_value_min=None, clip_value_max=None)
        pred_ctr_x = dx
        pred_ctr_y = dy
        pred_w = tensorlayerx.ops.exp(dw)
        pred_h = tensorlayerx.ops.exp(dh)
        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        x1 = tensorlayerx.reshape(x1, shape=-1)
        y1 = tensorlayerx.reshape(y1, shape=-1)
        x2 = tensorlayerx.reshape(x2, shape=-1)
        y2 = tensorlayerx.reshape(y2, shape=-1)
        return tensorlayerx.concat([x1, y1, x2, y2])

    def get_prediction(self, score, delta):
        bbox_prob = tensorlayerx.ops.softmax(score)
        return delta, bbox_prob

    def get_head(self):
        return self.head

    def get_assigned_targets(self):
        return self.assigned_targets

    def get_assigned_rois(self):
        return self.assigned_rois
