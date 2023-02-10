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
from models.bbox_utils import bbox2delta
from models.bbox_utils import delta2bbox
from models.heads.fcos_head import FCOSFeat
from core.workspace import register
__all__ = ['RetinaHead']


@register
class RetinaFeat(FCOSFeat):
    """We use FCOSFeat to construct conv layers in RetinaNet.
    We rename FCOSFeat to RetinaFeat to avoid confusion.
    """
    pass


@register
class RetinaHead(nn.Module):
    """Used in RetinaNet proposed in paper https://arxiv.org/pdf/1708.02002.pdf
    """
    __shared__ = ['num_classes']
    __inject__ = ['conv_feat', 'anchor_generator', 'bbox_assigner',
        'loss_class', 'loss_bbox', 'nms']

    def __init__(self, num_classes=80, conv_feat='RetinaFeat',
        anchor_generator='RetinaAnchorGenerator', bbox_assigner=\
        'MaxIoUAssigner', loss_class='FocalLoss', loss_bbox='SmoothL1Loss',
        nms='MultiClassNMS', prior_prob=0.01, nms_pre=1000, weights=[1.0, 
        1.0, 1.0, 1.0]):
        super(RetinaHead, self).__init__()
        self.num_classes = num_classes
        self.conv_feat = conv_feat
        self.anchor_generator = anchor_generator
        self.bbox_assigner = bbox_assigner
        self.loss_class = loss_class
        self.loss_bbox = loss_bbox
        self.nms = nms
        self.nms_pre = nms_pre
        self.weights = weights
        bias_init_value = -math.log((1 - prior_prob) / prior_prob)
        num_anchors = self.anchor_generator.num_anchors
        self.retina_cls = nn.GroupConv2d(in_channels=self.conv_feat.
            feat_out, out_channels=self.num_classes * num_anchors,
            kernel_size=3, stride=1, padding=1, W_init=xavier_uniform(),
            b_init=xavier_uniform(), data_format='channels_first')
        self.retina_reg = nn.GroupConv2d(in_channels=self.conv_feat.
            feat_out, out_channels=4 * num_anchors, kernel_size=3, stride=1,
            padding=1, W_init=xavier_uniform(), b_init=xavier_uniform(),
            data_format='channels_first')

    def forward(self, neck_feats, targets=None):
        cls_logits_list = []
        bboxes_reg_list = []
        for neck_feat in neck_feats:
            conv_cls_feat, conv_reg_feat = self.conv_feat(neck_feat)
            cls_logits = self.retina_cls(conv_cls_feat)
            bbox_reg = self.retina_reg(conv_reg_feat)
            cls_logits_list.append(cls_logits)
            bboxes_reg_list.append(bbox_reg)
        if self.training:
            return self.get_loss([cls_logits_list, bboxes_reg_list], targets)
        else:
            return [cls_logits_list, bboxes_reg_list]

    def get_loss(self, head_outputs, targets):
        """Here we calculate loss for a batch of images.
        We assign anchors to gts in each image and gather all the assigned
        postive and negative samples. Then loss is calculated on the gathered
        samples.
        """
        cls_logits_list, bboxes_reg_list = head_outputs
        anchors = self.anchor_generator(cls_logits_list)
        anchors = tensorlayerx.concat(anchors)
        matches_list, match_labels_list = [], []
        for gt_bbox in targets['gt_bbox']:
            matches, match_labels = self.bbox_assigner(anchors, gt_bbox)
            matches_list.append(matches)
            match_labels_list.append(match_labels)
        cls_logits = [_.transpose([0, 2, 3, 1]).reshape([0, -1, self.
            num_classes]) for _ in cls_logits_list]
        bboxes_reg = [_.transpose([0, 2, 3, 1]).reshape([0, -1, 4]) for _ in
            bboxes_reg_list]
        cls_logits = tensorlayerx.concat(cls_logits, axis=1)
        bboxes_reg = tensorlayerx.concat(bboxes_reg, axis=1)
        cls_pred_list, cls_tar_list = [], []
        reg_pred_list, reg_tar_list = [], []
        for matches, match_labels, cls_logit, bbox_reg, gt_bbox, gt_class in zip(
            matches_list, match_labels_list, cls_logits, bboxes_reg,
            targets['gt_bbox'], targets['gt_class']):
            pos_mask = match_labels == 1
            neg_mask = match_labels == 0
            chosen_mask = tensorlayerx.ops.logical_or(pos_mask, neg_mask)
            gt_class = gt_class.reshape([-1])
            bg_class = tensorlayerx.convert_to_tensor([self.num_classes],
                dtype=gt_class.dtype)
            gt_class = tensorlayerx.concat([gt_class, bg_class], axis=-1)
            matches = tensorlayerx.where(neg_mask, tensorlayerx.constant(x=\
                matches, fill_value=gt_class.size - 1, dtype=matches.dtype),
                matches)
            cls_pred = cls_logit[chosen_mask]
            cls_tar = gt_class[matches[chosen_mask]]
            reg_pred = bbox_reg[pos_mask].reshape([-1, 4])
            reg_tar = gt_bbox[matches[pos_mask]].reshape([-1, 4])
            reg_tar = bbox2delta(anchors[pos_mask], reg_tar, self.weights)
            cls_pred_list.append(cls_pred)
            cls_tar_list.append(cls_tar)
            reg_pred_list.append(reg_pred)
            reg_tar_list.append(reg_tar)
        cls_pred = tensorlayerx.concat(cls_pred_list)
        cls_tar = tensorlayerx.concat(cls_tar_list)
        reg_pred = tensorlayerx.concat(reg_pred_list)
        reg_tar = tensorlayerx.concat(reg_tar_list)
        avg_factor = max(1.0, reg_pred.shape[0])
        cls_loss = self.loss_class(cls_pred, cls_tar, reduction='sum'
            ) / avg_factor
        if reg_pred.shape[0] == 0:
            reg_loss = tensorlayerx.zeros([1])
            reg_loss.stop_gradient = False
        else:
            reg_loss = self.loss_bbox(reg_pred, reg_tar, reduction='sum'
                ) / avg_factor
        loss = cls_loss + reg_loss
        out_dict = {'loss_cls': cls_loss, 'loss_reg': reg_loss, 'loss': loss}
        return out_dict

    def get_bboxes_single(self, anchors, cls_scores_list, bbox_preds_list,
        im_shape, scale_factor, rescale=True):
        assert len(cls_scores_list) == len(bbox_preds_list)
        mlvl_bboxes = []
        mlvl_scores = []
        for anchor, cls_score, bbox_pred in zip(anchors, cls_scores_list,
            bbox_preds_list):
            cls_score = cls_score.reshape([-1, self.num_classes])
            bbox_pred = bbox_pred.reshape([-1, 4])
            if self.nms_pre is not None and cls_score.shape[0] > self.nms_pre:
                max_score = cls_score.max(axis=1)
                _, topk_inds = max_score.topk(self.nms_pre)
                bbox_pred = bbox_pred.gather(topk_inds)
                anchor = anchor.gather(topk_inds)
                cls_score = cls_score.gather(topk_inds)
            bbox_pred = delta2bbox(bbox_pred, anchor, self.weights).squeeze()
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(tensorlayerx.ops.sigmoid(cls_score))
        mlvl_bboxes = tensorlayerx.concat(mlvl_bboxes)
        mlvl_bboxes = tensorlayerx.ops.squeeze(mlvl_bboxes)
        if rescale:
            aa = tensorlayerx.concat([scale_factor[::-1], scale_factor[::-1]])
            mlvl_bboxes = mlvl_bboxes / aa
        mlvl_scores = tensorlayerx.concat(mlvl_scores)
        mlvl_scores = mlvl_scores.transpose([1, 0])
        return mlvl_bboxes, mlvl_scores

    def decode(self, anchors, cls_logits, bboxes_reg, im_shape, scale_factor):
        batch_bboxes = []
        batch_scores = []
        for img_id in range(cls_logits[0].shape[0]):
            num_lvls = len(cls_logits)
            cls_scores_list = [cls_logits[i][img_id] for i in range(num_lvls)]
            bbox_preds_list = [bboxes_reg[i][img_id] for i in range(num_lvls)]
            bboxes, scores = self.get_bboxes_single(anchors,
                cls_scores_list, bbox_preds_list, im_shape[img_id],
                scale_factor[img_id])
            batch_bboxes.append(bboxes)
            batch_scores.append(scores)
        batch_bboxes = tensorlayerx.ops.stack(batch_bboxes, axis=0)
        batch_scores = tensorlayerx.ops.stack(batch_scores, axis=0)
        return batch_bboxes, batch_scores

    def post_process(self, head_outputs, im_shape, scale_factor):
        cls_logits_list, bboxes_reg_list = head_outputs
        anchors = self.anchor_generator(cls_logits_list)
        cls_logits = [_.transpose([0, 2, 3, 1]) for _ in cls_logits_list]
        bboxes_reg = [_.transpose([0, 2, 3, 1]) for _ in bboxes_reg_list]
        bboxes, scores = self.decode(anchors, cls_logits, bboxes_reg,
            im_shape, scale_factor)
        bbox_pred, bbox_num, _ = self.nms(bboxes, scores)
        return bbox_pred, bbox_num
