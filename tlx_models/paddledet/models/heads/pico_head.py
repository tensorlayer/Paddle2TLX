from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import math
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import xavier_uniform
from tensorlayerx.nn.initializers import random_normal
from tensorlayerx.nn.initializers import Constant
from models.ops import get_static_shape
from ..assigners.utils import generate_anchors_for_grid_cell
from ..bbox_utils import bbox_center
from ..bbox_utils import batch_distance2bbox
from ..bbox_utils import bbox2distance
from core.workspace import register
from models.layers import ConvNormLayer
from .simota_head import OTAVFLHead
from .gfl_head import GFLHead
from models.necks.csp_pan import DPModule
eps = 1e-09
__all__ = ['PicoHead', 'PicoHeadV2', 'PicoFeat']


class PicoSE(nn.Module):

    def __init__(self, feat_channels):
        super(PicoSE, self).__init__()
        self.fc = nn.GroupConv2d(in_channels=feat_channels, out_channels=\
            feat_channels, kernel_size=1, padding=0, data_format=\
            'channels_first')
        self.conv = ConvNormLayer(feat_channels, feat_channels, 1, 1)

    def forward(self, feat, avg_feat):
        weight = tensorlayerx.ops.sigmoid(self.fc(avg_feat))
        out = self.conv(feat * weight)
        return out


@register
class PicoFeat(nn.Module):
    """
    PicoFeat of PicoDet

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the LiteGFLFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
        share_cls_reg (bool): Whether to share the cls and reg output.
        act (str): The act of per layers.
        use_se (bool): Whether to use se module.
    """

    def __init__(self, feat_in=256, feat_out=96, num_fpn_stride=3,
        num_convs=2, norm_type='bn', share_cls_reg=False, act='hard_swish',
        use_se=False):
        super(PicoFeat, self).__init__()
        self.num_convs = num_convs
        self.norm_type = norm_type
        self.share_cls_reg = share_cls_reg
        self.act = act
        self.use_se = use_se
        self.cls_convs = []
        self.reg_convs = []
        if use_se:
            assert share_cls_reg == True, 'In the case of using se, share_cls_reg must be set to True'
            self.se = nn.ModuleList()
        for stage_idx in range(num_fpn_stride):
            cls_subnet_convs = []
            reg_subnet_convs = []
            for i in range(self.num_convs):
                in_c = feat_in if i == 0 else feat_out
                cls_conv_dw = self.add_sublayer('cls_conv_dw{}.{}'.format(
                    stage_idx, i), ConvNormLayer(ch_in=in_c, ch_out=\
                    feat_out, filter_size=5, stride=1, groups=feat_out,
                    norm_type=norm_type, bias_on=False, lr_scale=2.0))
                cls_subnet_convs.append(cls_conv_dw)
                cls_conv_pw = self.add_sublayer('cls_conv_pw{}.{}'.format(
                    stage_idx, i), ConvNormLayer(ch_in=in_c, ch_out=\
                    feat_out, filter_size=1, stride=1, norm_type=norm_type,
                    bias_on=False, lr_scale=2.0))
                cls_subnet_convs.append(cls_conv_pw)
                if not self.share_cls_reg:
                    reg_conv_dw = self.add_sublayer('reg_conv_dw{}.{}'.
                        format(stage_idx, i), ConvNormLayer(ch_in=in_c,
                        ch_out=feat_out, filter_size=5, stride=1, groups=\
                        feat_out, norm_type=norm_type, bias_on=False,
                        lr_scale=2.0))
                    reg_subnet_convs.append(reg_conv_dw)
                    reg_conv_pw = self.add_sublayer('reg_conv_pw{}.{}'.
                        format(stage_idx, i), ConvNormLayer(ch_in=in_c,
                        ch_out=feat_out, filter_size=1, stride=1, norm_type
                        =norm_type, bias_on=False, lr_scale=2.0))
                    reg_subnet_convs.append(reg_conv_pw)
            self.cls_convs.append(cls_subnet_convs)
            self.reg_convs.append(reg_subnet_convs)
            if use_se:
                self.se.append(PicoSE(feat_out))

    def act_func(self, x):
        if self.act == 'leaky_relu':
            x = tensorlayerx.ops.leaky_relu(x)
        elif self.act == 'hard_swish':
            x = paddle.nn.functional.hardswish(x)
        elif self.act == 'relu6':
            x = tensorlayerx.nn.ReLU6()(x)
        return x

    def forward(self, fpn_feat, stage_idx):
        assert stage_idx < len(self.cls_convs)
        cls_feat = fpn_feat
        reg_feat = fpn_feat
        for i in range(len(self.cls_convs[stage_idx])):
            cls_feat = self.act_func(self.cls_convs[stage_idx][i](cls_feat))
            reg_feat = cls_feat
            if not self.share_cls_reg:
                reg_feat = self.act_func(self.reg_convs[stage_idx][i](reg_feat)
                    )
        if self.use_se:
            avg_feat = paddle.nn.functional.adaptive_avg_pool2d(cls_feat, (
                1, 1))
            se_feat = self.act_func(self.se[stage_idx](cls_feat, avg_feat))
            return cls_feat, se_feat
        return cls_feat, reg_feat


@register
class PicoHead(OTAVFLHead):
    """
    PicoHead
    Args:
        conv_feat (object): Instance of 'PicoFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_class (object): Instance of VariFocalLoss.
        loss_dfl (object): Instance of DistributionFocalLoss.
        loss_bbox (object): Instance of bbox loss.
        assigner (object): Instance of label assigner.
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 7.
    """
    __inject__ = ['conv_feat', 'dgqp_module', 'loss_class', 'loss_dfl',
        'loss_bbox', 'assigner', 'nms']
    __shared__ = ['num_classes', 'eval_size']

    def __init__(self, conv_feat='PicoFeat', dgqp_module=None, num_classes=\
        80, fpn_stride=[8, 16, 32], prior_prob=0.01, loss_class=\
        'VariFocalLoss', loss_dfl='DistributionFocalLoss', loss_bbox=\
        'GIoULoss', assigner='SimOTAAssigner', reg_max=16, feat_in_chan=96,
        nms=None, nms_pre=1000, cell_offset=0, eval_size=None):
        super(PicoHead, self).__init__(conv_feat=conv_feat, dgqp_module=\
            dgqp_module, num_classes=num_classes, fpn_stride=fpn_stride,
            prior_prob=prior_prob, loss_class=loss_class, loss_dfl=loss_dfl,
            loss_bbox=loss_bbox, assigner=assigner, reg_max=reg_max,
            feat_in_chan=feat_in_chan, nms=nms, nms_pre=nms_pre,
            cell_offset=cell_offset)
        self.conv_feat = conv_feat
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_vfl = loss_class
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox
        self.assigner = assigner
        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.eval_size = eval_size
        self.use_sigmoid = self.loss_vfl.use_sigmoid
        if self.use_sigmoid:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self.gfl_head_cls = None
        self.gfl_head_reg = None
        self.scales_regs = None
        self.head_cls_list = []
        self.head_reg_list = []
        for i in range(len(fpn_stride)):
            head_cls = self.add_sublayer('head_cls' + str(i), nn.
                GroupConv2d(in_channels=self.feat_in_chan, out_channels=\
                self.cls_out_channels + 4 * (self.reg_max + 1) if self.
                conv_feat.share_cls_reg else self.cls_out_channels,
                kernel_size=1, stride=1, padding=0, W_init=xavier_uniform(),
                b_init=xavier_uniform(), data_format='channels_first'))
            self.head_cls_list.append(head_cls)
            if not self.conv_feat.share_cls_reg:
                head_reg = self.add_sublayer('head_reg' + str(i), nn.
                    GroupConv2d(in_channels=self.feat_in_chan, out_channels
                    =4 * (self.reg_max + 1), kernel_size=1, stride=1,
                    padding=0, W_init=xavier_uniform(), b_init=\
                    xavier_uniform(), data_format='channels_first'))
                self.head_reg_list.append(head_reg)
        if self.eval_size:
            self.anchor_points, self.stride_tensor = self._generate_anchors()

    def forward(self, fpn_feats, export_post_process=True):
        assert len(fpn_feats) == len(self.fpn_stride
            ), 'The size of fpn_feats is not equal to size of fpn_stride'
        if self.training:
            return self.forward_train(fpn_feats)
        else:
            return self.forward_eval(fpn_feats, export_post_process=\
                export_post_process)

    def forward_train(self, fpn_feats):
        cls_logits_list, bboxes_reg_list = [], []
        for i, fpn_feat in enumerate(fpn_feats):
            conv_cls_feat, conv_reg_feat = self.conv_feat(fpn_feat, i)
            if self.conv_feat.share_cls_reg:
                cls_logits = self.head_cls_list[i](conv_cls_feat)
                cls_score, bbox_pred = tensorlayerx.ops.split(cls_logits, [
                    self.cls_out_channels, 4 * (self.reg_max + 1)], axis=1)
            else:
                cls_score = self.head_cls_list[i](conv_cls_feat)
                bbox_pred = self.head_reg_list[i](conv_reg_feat)
            if self.dgqp_module:
                quality_score = self.dgqp_module(bbox_pred)
                cls_score = F.sigmoid(cls_score) * quality_score
            cls_logits_list.append(cls_score)
            bboxes_reg_list.append(bbox_pred)
        return cls_logits_list, bboxes_reg_list

    def forward_eval(self, fpn_feats, export_post_process=True):
        if self.eval_size:
            anchor_points, stride_tensor = (self.anchor_points, self.
                stride_tensor)
        else:
            anchor_points, stride_tensor = self._generate_anchors(fpn_feats)
        cls_logits_list, bboxes_reg_list = [], []
        for i, fpn_feat in enumerate(fpn_feats):
            conv_cls_feat, conv_reg_feat = self.conv_feat(fpn_feat, i)
            if self.conv_feat.share_cls_reg:
                cls_logits = self.head_cls_list[i](conv_cls_feat)
                cls_score, bbox_pred = tensorlayerx.ops.split(cls_logits, [
                    self.cls_out_channels, 4 * (self.reg_max + 1)], axis=1)
            else:
                cls_score = self.head_cls_list[i](conv_cls_feat)
                bbox_pred = self.head_reg_list[i](conv_reg_feat)
            if self.dgqp_module:
                quality_score = self.dgqp_module(bbox_pred)
                cls_score = F.sigmoid(cls_score) * quality_score
            if not export_post_process:
                cls_score_out = tensorlayerx.ops.sigmoid(cls_score).reshape([
                    1, self.cls_out_channels, -1]).transpose([0, 2, 1])
                bbox_pred = bbox_pred.reshape([1, (self.reg_max + 1) * 4, -1]
                    ).transpose([0, 2, 1])
            else:
                b, _, h, w = fpn_feat.shape
                l = h * w
                cls_score_out = tensorlayerx.ops.sigmoid(cls_score.reshape(
                    [b, self.cls_out_channels, l]))
                bbox_pred = bbox_pred.transpose([0, 2, 3, 1])
                bbox_pred = self.distribution_project(bbox_pred)
                bbox_pred = bbox_pred.reshape([b, l, 4])
            cls_logits_list.append(cls_score_out)
            bboxes_reg_list.append(bbox_pred)
        if export_post_process:
            cls_logits_list = tensorlayerx.concat(cls_logits_list, axis=-1)
            bboxes_reg_list = tensorlayerx.concat(bboxes_reg_list, axis=1)
            bboxes_reg_list = batch_distance2bbox(anchor_points,
                bboxes_reg_list)
            bboxes_reg_list *= stride_tensor
        return cls_logits_list, bboxes_reg_list

    def _generate_anchors(self, feats=None):
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_stride):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = math.ceil(self.eval_size[0] / stride)
                w = math.ceil(self.eval_size[1] / stride)
            aa = tensorlayerx.ops.arange(start=0, limit=w)
            bb = tensorlayerx.ops.arange(start=0, limit=h)
            shift_x = aa + self.cell_offset
            shift_y = bb + self.cell_offset
            shift_y, shift_x = tensorlayerx.meshgrid(shift_y, shift_x)
            anchor_point = tensorlayerx.cast(tensorlayerx.ops.stack([
                shift_x, shift_y], axis=-1), dtype='float32')
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(tensorlayerx.constant(shape=[h * w, 1],
                dtype='float32', value=stride))
        anchor_points = tensorlayerx.concat(anchor_points)
        stride_tensor = tensorlayerx.concat(stride_tensor)
        return anchor_points, stride_tensor

    def post_process(self, head_outs, scale_factor, export_nms=True):
        pred_scores, pred_bboxes = head_outs
        if not export_nms:
            return pred_bboxes, pred_scores
        else:
            scale_y, scale_x = tensorlayerx.ops.split(scale_factor, 2, axis=-1)
            scale_factor = tensorlayerx.concat([scale_x, scale_y, scale_x,
                scale_y], axis=-1).reshape([-1, 1, 4])
            pred_bboxes /= scale_factor
            bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num


@register
class PicoHeadV2(GFLHead):
    """
    PicoHeadV2
    Args:
        conv_feat (object): Instance of 'PicoFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_class (object): Instance of VariFocalLoss.
        loss_dfl (object): Instance of DistributionFocalLoss.
        loss_bbox (object): Instance of bbox loss.
        assigner (object): Instance of label assigner.
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 7.
    """
    __inject__ = ['conv_feat', 'dgqp_module', 'loss_class', 'loss_dfl',
        'loss_bbox', 'static_assigner', 'assigner', 'nms']
    __shared__ = ['num_classes', 'eval_size']

    def __init__(self, conv_feat='PicoFeatV2', dgqp_module=None,
        num_classes=80, fpn_stride=[8, 16, 32], prior_prob=0.01,
        use_align_head=True, loss_class='VariFocalLoss', loss_dfl=\
        'DistributionFocalLoss', loss_bbox='GIoULoss',
        static_assigner_epoch=60, static_assigner='ATSSAssigner', assigner=\
        'TaskAlignedAssigner', reg_max=16, feat_in_chan=96, nms=None,
        nms_pre=1000, cell_offset=0, act='hard_swish', grid_cell_scale=5.0,
        eval_size=None):
        super(PicoHeadV2, self).__init__(conv_feat=conv_feat, dgqp_module=\
            dgqp_module, num_classes=num_classes, fpn_stride=fpn_stride,
            prior_prob=prior_prob, loss_class=loss_class, loss_dfl=loss_dfl,
            loss_bbox=loss_bbox, reg_max=reg_max, feat_in_chan=feat_in_chan,
            nms=nms, nms_pre=nms_pre, cell_offset=cell_offset)
        self.conv_feat = conv_feat
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_vfl = loss_class
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox
        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.act = act
        self.grid_cell_scale = grid_cell_scale
        self.use_align_head = use_align_head
        self.cls_out_channels = self.num_classes
        self.eval_size = eval_size
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self.gfl_head_cls = None
        self.gfl_head_reg = None
        self.scales_regs = None
        self.head_cls_list = []
        self.head_reg_list = []
        self.cls_align = nn.ModuleList()
        for i in range(len(fpn_stride)):
            head_cls = self.add_sublayer('head_cls' + str(i), nn.
                GroupConv2d(in_channels=self.feat_in_chan, out_channels=\
                self.cls_out_channels, kernel_size=1, stride=1, padding=0,
                W_init=xavier_uniform(), b_init=xavier_uniform(),
                data_format='channels_first'))
            self.head_cls_list.append(head_cls)
            head_reg = self.add_sublayer('head_reg' + str(i), nn.
                GroupConv2d(in_channels=self.feat_in_chan, out_channels=4 *
                (self.reg_max + 1), kernel_size=1, stride=1, padding=0,
                W_init=xavier_uniform(), b_init=xavier_uniform(),
                data_format='channels_first'))
            self.head_reg_list.append(head_reg)
            if self.use_align_head:
                self.cls_align.append(DPModule(self.feat_in_chan, 1, 5, act
                    =self.act, use_act_in_out=False))
        if self.eval_size:
            self.anchor_points, self.stride_tensor = self._generate_anchors()

    def forward(self, fpn_feats, export_post_process=True):
        assert len(fpn_feats) == len(self.fpn_stride
            ), 'The size of fpn_feats is not equal to size of fpn_stride'
        if self.training:
            return self.forward_train(fpn_feats)
        else:
            return self.forward_eval(fpn_feats, export_post_process=\
                export_post_process)

    def forward_train(self, fpn_feats):
        cls_score_list, reg_list, box_list = [], [], []
        for i, (fpn_feat, stride) in enumerate(zip(fpn_feats, self.fpn_stride)
            ):
            b, _, h, w = get_static_shape(fpn_feat)
            conv_cls_feat, se_feat = self.conv_feat(fpn_feat, i)
            cls_logit = self.head_cls_list[i](se_feat)
            reg_pred = self.head_reg_list[i](se_feat)
            if self.use_align_head:
                cls_prob = tensorlayerx.ops.sigmoid(self.cls_align[i](
                    conv_cls_feat))
                cls_logit_sigmoid = tensorlayerx.ops.sigmoid(cls_logit)
                cls_score = (cls_logit_sigmoid * cls_prob + eps).sqrt()
            else:
                cls_score = tensorlayerx.ops.sigmoid(cls_logit)
            cls_score_out = cls_score.transpose([0, 2, 3, 1])
            bbox_pred = reg_pred.transpose([0, 2, 3, 1])
            b, cell_h, cell_w, _ = (paddle2tlx.pd2tlx.ops.tlxops.
                tlx_get_tensor_shape(cls_score_out))
            y, x = self.get_single_level_center_point([cell_h, cell_w],
                stride, cell_offset=self.cell_offset)
            center_points = tensorlayerx.ops.stack([x, y], axis=-1)
            cls_score_out = cls_score_out.reshape([b, -1, self.
                cls_out_channels])
            bbox_pred = self.distribution_project(bbox_pred) * stride
            bbox_pred = bbox_pred.reshape([b, cell_h * cell_w, 4])
            bbox_pred = batch_distance2bbox(center_points, bbox_pred,
                max_shapes=None)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_list.append(reg_pred.flatten(2).transpose([0, 2, 1]))
            box_list.append(bbox_pred / stride)
        cls_score_list = tensorlayerx.concat(cls_score_list, axis=1)
        box_list = tensorlayerx.concat(box_list, axis=1)
        reg_list = tensorlayerx.concat(reg_list, axis=1)
        return cls_score_list, reg_list, box_list, fpn_feats

    def forward_eval(self, fpn_feats, export_post_process=True):
        if self.eval_size:
            anchor_points, stride_tensor = (self.anchor_points, self.
                stride_tensor)
        else:
            anchor_points, stride_tensor = self._generate_anchors(fpn_feats)
        cls_score_list, box_list = [], []
        for i, (fpn_feat, stride) in enumerate(zip(fpn_feats, self.fpn_stride)
            ):
            b, _, h, w = fpn_feat.shape
            conv_cls_feat, se_feat = self.conv_feat(fpn_feat, i)
            cls_logit = self.head_cls_list[i](se_feat)
            reg_pred = self.head_reg_list[i](se_feat)
            if self.use_align_head:
                cls_prob = tensorlayerx.ops.sigmoid(self.cls_align[i](
                    conv_cls_feat))
                cls_logit_sigmoid = tensorlayerx.ops.sigmoid(cls_logit)
                cls_score = (cls_logit_sigmoid * cls_prob + eps).sqrt()
            else:
                cls_score = tensorlayerx.ops.sigmoid(cls_logit)
            if not export_post_process:
                cls_score_list.append(cls_score.reshape([1, self.
                    cls_out_channels, -1]).transpose([0, 2, 1]))
                box_list.append(reg_pred.reshape([1, (self.reg_max + 1) * 4,
                    -1]).transpose([0, 2, 1]))
            else:
                l = h * w
                cls_score_out = cls_score.reshape([b, self.cls_out_channels, l]
                    )
                bbox_pred = reg_pred.transpose([0, 2, 3, 1])
                bbox_pred = self.distribution_project(bbox_pred)
                bbox_pred = bbox_pred.reshape([b, l, 4])
                cls_score_list.append(cls_score_out)
                box_list.append(bbox_pred)
        if export_post_process:
            cls_score_list = tensorlayerx.concat(cls_score_list, axis=-1)
            box_list = tensorlayerx.concat(box_list, axis=1)
            box_list = batch_distance2bbox(anchor_points, box_list)
            box_list *= stride_tensor
        return cls_score_list, box_list

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_regs, pred_bboxes, fpn_feats = head_outs
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None
        num_imgs = gt_meta['im_id'].shape[0]
        pad_gt_mask = gt_meta['pad_gt_mask']
        anchors, _, num_anchors_list, stride_tensor_list = (
            generate_anchors_for_grid_cell(fpn_feats, self.fpn_stride, self
            .grid_cell_scale, self.cell_offset))
        centers = bbox_center(anchors)
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = (self.
                static_assigner(anchors, num_anchors_list, gt_labels,
                gt_bboxes, pad_gt_mask, bg_index=self.num_classes,
                gt_scores=gt_scores, pred_bboxes=pred_bboxes.detach() *
                stride_tensor_list))
        else:
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                pred_scores.detach(), pred_bboxes.detach() *
                stride_tensor_list, centers, num_anchors_list, gt_labels,
                gt_bboxes, pad_gt_mask, bg_index=self.num_classes,
                gt_scores=gt_scores)
        assigned_bboxes /= stride_tensor_list
        centers_shape = centers.shape
        flatten_centers = centers.expand([num_imgs, centers_shape[0],
            centers_shape[1]]).reshape([-1, 2])
        flatten_strides = stride_tensor_list.expand([num_imgs,
            centers_shape[0], 1]).reshape([-1, 1])
        flatten_cls_preds = pred_scores.reshape([-1, self.num_classes])
        flatten_regs = pred_regs.reshape([-1, 4 * (self.reg_max + 1)])
        flatten_bboxes = pred_bboxes.reshape([-1, 4])
        flatten_bbox_targets = assigned_bboxes.reshape([-1, 4])
        flatten_labels = assigned_labels.reshape([-1])
        flatten_assigned_scores = assigned_scores.reshape([-1, self.
            num_classes])
        pos_inds = paddle2tlx.pd2tlx.ops.tlxops.tlx_nonzero(tensorlayerx.
            ops.logical_and(flatten_labels >= 0, flatten_labels < self.
            num_classes), as_tuple=False).squeeze(1)
        num_total_pos = len(pos_inds)
        if num_total_pos > 0:
            pos_bbox_targets = tensorlayerx.gather(flatten_bbox_targets,
                pos_inds, axis=0)
            pos_decode_bbox_pred = tensorlayerx.gather(flatten_bboxes,
                pos_inds, axis=0)
            pos_reg = tensorlayerx.gather(flatten_regs, pos_inds, axis=0)
            pos_strides = tensorlayerx.gather(flatten_strides, pos_inds, axis=0
                )
            aa = tensorlayerx.gather(flatten_centers, pos_inds, axis=0)
            pos_centers = aa / pos_strides
            weight_targets = flatten_assigned_scores.detach()
            weight_targets = tensorlayerx.gather(weight_targets.max(axis=1,
                keepdim=True), pos_inds, axis=0)
            pred_corners = pos_reg.reshape([-1, self.reg_max + 1])
            target_corners = bbox2distance(pos_centers, pos_bbox_targets,
                self.reg_max).reshape([-1])
            loss_bbox = tensorlayerx.reduce_sum(self.loss_bbox(
                pos_decode_bbox_pred, pos_bbox_targets) * weight_targets)
            loss_dfl = self.loss_dfl(pred_corners, target_corners, weight=\
                weight_targets.expand([-1, 4]).reshape([-1]), avg_factor=4.0)
        else:
            loss_bbox = tensorlayerx.zeros([1])
            loss_dfl = tensorlayerx.zeros([1])
        avg_factor = flatten_assigned_scores.sum()
        loss_vfl = self.loss_vfl(flatten_cls_preds, flatten_assigned_scores,
            avg_factor=avg_factor)
        loss_bbox = loss_bbox / avg_factor
        loss_dfl = loss_dfl / avg_factor
        loss_states = dict(loss_vfl=loss_vfl, loss_bbox=loss_bbox, loss_dfl
            =loss_dfl)
        return loss_states

    def _generate_anchors(self, feats=None):
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_stride):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = math.ceil(self.eval_size[0] / stride)
                w = math.ceil(self.eval_size[1] / stride)
            aa = tensorlayerx.ops.arange(start=0, limit=w)
            bb = tensorlayerx.ops.arange(start=0, limit=h)
            shift_x = aa + self.cell_offset
            shift_y = bb + self.cell_offset
            shift_y, shift_x = tensorlayerx.meshgrid(shift_y, shift_x)
            anchor_point = tensorlayerx.cast(tensorlayerx.ops.stack([
                shift_x, shift_y], axis=-1), dtype='float32')
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(tensorlayerx.constant(shape=[h * w, 1],
                dtype='float32', value=stride))
        anchor_points = tensorlayerx.concat(anchor_points)
        stride_tensor = tensorlayerx.concat(stride_tensor)
        return anchor_points, stride_tensor

    def post_process(self, head_outs, scale_factor, export_nms=True):
        pred_scores, pred_bboxes = head_outs
        if not export_nms:
            return pred_bboxes, pred_scores
        else:
            scale_y, scale_x = tensorlayerx.ops.split(scale_factor, 2, axis=-1)
            scale_factor = tensorlayerx.concat([scale_x, scale_y, scale_x,
                scale_y], axis=-1).reshape([-1, 1, 4])
            pred_bboxes /= scale_factor
            bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num
