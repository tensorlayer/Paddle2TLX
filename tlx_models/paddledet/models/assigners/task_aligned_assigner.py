from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from core.workspace import register
from ..bbox_utils import batch_iou_similarity
from .utils import gather_topk_anchors
from .utils import check_points_inside_bboxes
from .utils import compute_max_iou_anchor
__all__ = ['TaskAlignedAssigner']


@register
class TaskAlignedAssigner(nn.Module):
    """TOOD: Task-aligned One-stage Object Detection
    """

    def __init__(self, topk=13, alpha=1.0, beta=6.0, eps=1e-09):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred_scores, pred_bboxes, anchor_points,
        num_anchors_list, gt_labels, gt_bboxes, pad_gt_mask, bg_index,
        gt_scores=None):
        """This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector
           only can predict positive distance)
        4. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            pred_scores (Tensor, float32): predicted class probability, shape(B, L, C)
            pred_bboxes (Tensor, float32): predicted bounding boxes, shape(B, L, 4)
            anchor_points (Tensor, float32): pre-defined anchors, shape(L, 2), "cxcy" format
            num_anchors_list (List): num of anchors in each level, shape(L)
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes, shape(B, n, 1)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C)
        """
        assert pred_scores.ndim == pred_bboxes.ndim
        assert gt_labels.ndim == gt_bboxes.ndim and gt_bboxes.ndim == 3
        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_bboxes.shape
        if num_max_boxes == 0:
            assigned_labels = tensorlayerx.constant(shape=[batch_size,
                num_anchors], dtype='int32', value=bg_index)
            assigned_bboxes = tensorlayerx.zeros([batch_size, num_anchors, 4])
            assigned_scores = tensorlayerx.zeros([batch_size, num_anchors,
                num_classes])
            return assigned_labels, assigned_bboxes, assigned_scores
        ious = batch_iou_similarity(gt_bboxes, pred_bboxes)
        pred_scores = pred_scores.transpose([0, 2, 1])
        batch_ind = tensorlayerx.ops.arange(start=0, dtype=gt_labels.dtype,
            limit=batch_size).unsqueeze(-1)
        gt_labels_ind = tensorlayerx.ops.stack([batch_ind.tile([1,
            num_max_boxes]), gt_labels.squeeze(-1)], axis=-1)
        bbox_cls_scores = tensorlayerx.gather_nd(pred_scores, gt_labels_ind)
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self
            .beta)
        is_in_gts = check_points_inside_bboxes(anchor_points, gt_bboxes)
        is_in_topk = gather_topk_anchors(alignment_metrics * is_in_gts,
            self.topk, topk_mask=pad_gt_mask)
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask
        mask_positive_sum = mask_positive.sum(axis=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile([
                1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = tensorlayerx.where(mask_multiple_gts,
                is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        assigned_gt_index = mask_positive.argmax(axis=-2)
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = tensorlayerx.gather(gt_labels.flatten(),
            assigned_gt_index.flatten(), axis=0)
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = tensorlayerx.where(mask_positive_sum > 0,
            assigned_labels, tensorlayerx.constant(x=assigned_labels,
            fill_value=bg_index, dtype=assigned_labels.dtype))
        assigned_bboxes = tensorlayerx.gather(gt_bboxes.reshape([-1, 4]),
            assigned_gt_index.flatten(), axis=0)
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])
        assigned_scores = tensorlayerx.ops.OneHot(assigned_labels, 
            num_classes + 1)
        ind = list(range(num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = tensorlayerx.index_select(assigned_scores,
            tensorlayerx.convert_to_tensor(ind), axis=-1)
        alignment_metrics *= mask_positive
        max_metrics_per_instance = alignment_metrics.max(axis=-1, keepdim=True)
        max_ious_per_instance = (ious * mask_positive).max(axis=-1, keepdim
            =True)
        alignment_metrics = alignment_metrics / (max_metrics_per_instance +
            self.eps) * max_ious_per_instance
        alignment_metrics = alignment_metrics.max(-2).unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics
        return assigned_labels, assigned_bboxes, assigned_scores
