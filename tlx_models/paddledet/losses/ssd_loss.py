from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from core.workspace import register
from models.bbox_utils import iou_similarity
from models.bbox_utils import bbox2delta
__all__ = ['SSDLoss']


@register
class SSDLoss(nn.Module):
    """
    SSDLoss

    Args:
        overlap_threshold (float32, optional): IoU threshold for negative bboxes
            and positive bboxes, 0.5 by default.
        neg_pos_ratio (float): The ratio of negative samples / positive samples.
        loc_loss_weight (float): The weight of loc_loss.
        conf_loss_weight (float): The weight of conf_loss.
        prior_box_var (list): Variances corresponding to prior box coord, [0.1,
            0.1, 0.2, 0.2] by default.
    """

    def __init__(self, overlap_threshold=0.5, neg_pos_ratio=3.0,
        loc_loss_weight=1.0, conf_loss_weight=1.0, prior_box_var=[0.1, 0.1,
        0.2, 0.2]):
        super(SSDLoss, self).__init__()
        self.overlap_threshold = overlap_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.loc_loss_weight = loc_loss_weight
        self.conf_loss_weight = conf_loss_weight
        self.prior_box_var = [(1.0 / a) for a in prior_box_var]

    def _bipartite_match_for_batch(self, gt_bbox, gt_label, prior_boxes,
        bg_index):
        """
        Args:
            gt_bbox (Tensor): [B, N, 4]
            gt_label (Tensor): [B, N, 1]
            prior_boxes (Tensor): [A, 4]
            bg_index (int): Background class index
        """
        batch_size, num_priors = gt_bbox.shape[0], prior_boxes.shape[0]
        ious = iou_similarity(gt_bbox.reshape((-1, 4)), prior_boxes).reshape((
            batch_size, -1, num_priors))
        prior_max_iou, prior_argmax_iou = ious.max(axis=1), ious.argmax(axis=1)
        gt_max_iou, gt_argmax_iou = ious.max(axis=2), ious.argmax(axis=2)
        batch_ind = tensorlayerx.ops.arange(start=0, dtype='int64', limit=\
            batch_size).unsqueeze(-1)
        prior_argmax_iou = tensorlayerx.ops.stack([batch_ind.tile([1,
            num_priors]), prior_argmax_iou], axis=-1)
        targets_bbox = tensorlayerx.gather_nd(gt_bbox, prior_argmax_iou)
        targets_label = tensorlayerx.gather_nd(gt_label, prior_argmax_iou)
        bg_index_tensor = tensorlayerx.constant(shape=[batch_size,
            num_priors, 1], dtype='int64', value=bg_index)
        targets_label = tensorlayerx.where(prior_max_iou.unsqueeze(-1) <
            self.overlap_threshold, bg_index_tensor, targets_label)
        batch_ind = (batch_ind * num_priors + gt_argmax_iou).flatten()
        targets_bbox = tensorlayerx.scatter_update(targets_bbox.reshape([-1,
            4]), batch_ind, gt_bbox.reshape([-1, 4])).reshape([batch_size, 
            -1, 4])
        targets_label = tensorlayerx.scatter_update(targets_label.reshape([
            -1, 1]), batch_ind, gt_label.reshape([-1, 1])).reshape([
            batch_size, -1, 1])
        targets_label[:, :1] = bg_index
        prior_boxes = prior_boxes.unsqueeze(0).tile([batch_size, 1, 1])
        targets_bbox = bbox2delta(prior_boxes.reshape([-1, 4]),
            targets_bbox.reshape([-1, 4]), self.prior_box_var)
        targets_bbox = targets_bbox.reshape([batch_size, -1, 4])
        return targets_bbox, targets_label

    def _mine_hard_example(self, conf_loss, targets_label, bg_index,
        mine_neg_ratio=0.01):
        pos = (targets_label != bg_index).astype(conf_loss.dtype)
        num_pos = pos.sum(axis=1, keepdim=True)
        neg = (targets_label == bg_index).astype(conf_loss.dtype)
        conf_loss = conf_loss.detach() * neg
        loss_idx = conf_loss.argsort(axis=1, descending=True)
        idx_rank = loss_idx.argsort(axis=1)
        num_negs = []
        for i in range(conf_loss.shape[0]):
            cur_num_pos = num_pos[i]
            num_neg = tensorlayerx.ops.clip_by_value(cur_num_pos * self.
                neg_pos_ratio, clip_value_max=pos.shape[1], clip_value_min=None
                )
            num_neg = (num_neg if num_neg > 0 else tensorlayerx.
                convert_to_tensor([pos.shape[1] * mine_neg_ratio]))
            num_negs.append(num_neg)
        num_negs = tensorlayerx.ops.stack(num_negs).expand_as(idx_rank)
        neg_mask = (idx_rank < num_negs).astype(conf_loss.dtype)
        return (neg_mask + pos).astype('bool')

    def forward(self, boxes, scores, gt_bbox, gt_label, prior_boxes):
        boxes = tensorlayerx.concat(boxes, axis=1)
        scores = tensorlayerx.concat(scores, axis=1)
        gt_label = gt_label.unsqueeze(-1).astype('int64')
        prior_boxes = tensorlayerx.concat(prior_boxes, axis=0)
        bg_index = scores.shape[-1] - 1
        targets_bbox, targets_label = self._bipartite_match_for_batch(gt_bbox,
            gt_label, prior_boxes, bg_index)
        targets_bbox.stop_gradient = True
        targets_label.stop_gradient = True
        bbox_mask = tensorlayerx.tile(targets_label != bg_index, [1, 1, 4])
        if bbox_mask.astype(boxes.dtype).sum() > 0:
            location = tensorlayerx.mask_select(boxes, bbox_mask)
            targets_bbox = tensorlayerx.mask_select(targets_bbox, bbox_mask)
            loc_loss = paddle.nn.functional.smooth_l1_loss(location,
                targets_bbox, reduction='sum')
            loc_loss = loc_loss * self.loc_loss_weight
        else:
            loc_loss = tensorlayerx.zeros([1])
        conf_loss = paddle2tlx.pd2tlx.ops.tlxops.tlx_cross_entropy(scores,
            targets_label, reduction='none')
        label_mask = self._mine_hard_example(conf_loss.squeeze(-1),
            targets_label.squeeze(-1), bg_index)
        conf_loss = tensorlayerx.mask_select(conf_loss, label_mask.
            unsqueeze(-1))
        conf_loss = conf_loss.sum() * self.conf_loss_weight
        normalizer = (targets_label != bg_index).astype('float32').sum().clip(
            min=1)
        loss = (conf_loss + loc_loss) / normalizer
        return loss
