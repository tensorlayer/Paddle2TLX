from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import numpy as np
import tensorlayerx
import tensorlayerx.nn as nn
from core.workspace import register
from ..bbox_utils import iou_similarity
from ..bbox_utils import batch_iou_similarity
from ..bbox_utils import bbox_center
from .utils import check_points_inside_bboxes
from .utils import compute_max_iou_anchor
from .utils import compute_max_iou_gt
__all__ = ['ATSSAssigner']


@register
class ATSSAssigner(nn.Module):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection
     via Adaptive Training Sample Selection
    """
    __shared__ = ['num_classes']

    def __init__(self, topk=9, num_classes=80, force_gt_matching=False, eps
        =1e-09):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching
        self.eps = eps

    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list,
        pad_gt_mask):
        gt2anchor_distances_list = tensorlayerx.ops.split(gt2anchor_distances,
            num_anchors_list, axis=-1)
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0] + num_anchors_index[:-1]
        is_in_topk_list = []
        topk_idxs_list = []
        for distances, anchors_index in zip(gt2anchor_distances_list,
            num_anchors_index):
            num_anchors = distances.shape[-1]
            _, topk_idxs = tensorlayerx.ops.topk(distances, self.topk, axis
                =-1, largest=False)
            topk_idxs_list.append(topk_idxs + anchors_index)
            is_in_topk = tensorlayerx.ops.OneHot(topk_idxs, num_anchors).sum(
                axis=-2).astype(gt2anchor_distances.dtype)
            is_in_topk_list.append(is_in_topk * pad_gt_mask)
        is_in_topk_list = tensorlayerx.concat(is_in_topk_list, axis=-1)
        topk_idxs_list = tensorlayerx.concat(topk_idxs_list, axis=-1)
        return is_in_topk_list, topk_idxs_list

    def forward(self, anchor_bboxes, num_anchors_list, gt_labels, gt_bboxes,
        pad_gt_mask, bg_index, gt_scores=None, pred_bboxes=None):
        """This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt
        7. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            anchor_bboxes (Tensor, float32): pre-defined anchors, shape(L, 4),
                    "xmin, xmax, ymin, ymax" format
            num_anchors_list (List): num of anchors in each level
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes,
                    shape(B, n, 1), if None, then it will initialize with one_hot label
            pred_bboxes (Tensor, float32, optional): predicted bounding boxes, shape(B, L, 4)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C), if pred_bboxes is not None, then output ious
        """
        assert gt_labels.ndim == gt_bboxes.ndim and gt_bboxes.ndim == 3
        num_anchors, _ = anchor_bboxes.shape
        batch_size, num_max_boxes, _ = gt_bboxes.shape
        if num_max_boxes == 0:
            assigned_labels = tensorlayerx.constant(shape=[batch_size,
                num_anchors], dtype='int32', value=bg_index)
            assigned_bboxes = tensorlayerx.zeros([batch_size, num_anchors, 4])
            assigned_scores = tensorlayerx.zeros([batch_size, num_anchors,
                self.num_classes])
            return assigned_labels, assigned_bboxes, assigned_scores
        ious = iou_similarity(gt_bboxes.reshape([-1, 4]), anchor_bboxes)
        ious = ious.reshape([batch_size, -1, num_anchors])
        gt_centers = bbox_center(gt_bboxes.reshape([-1, 4])).unsqueeze(1)
        anchor_centers = bbox_center(anchor_bboxes)
        gt2anchor_distances = (gt_centers - anchor_centers.unsqueeze(0)).norm(
            2, axis=-1).reshape([batch_size, -1, num_anchors])
        is_in_topk, topk_idxs = self._gather_topk_pyramid(gt2anchor_distances,
            num_anchors_list, pad_gt_mask)
        iou_candidates = ious * is_in_topk
        iou_threshold = paddle.index_sample(iou_candidates.flatten(
            stop_axis=-2), topk_idxs.flatten(stop_axis=-2))
        iou_threshold = iou_threshold.reshape([batch_size, num_max_boxes, -1])
        iou_threshold = iou_threshold.mean(axis=-1, keepdim=True
            ) + iou_threshold.std(axis=-1, keepdim=True)
        is_in_topk = tensorlayerx.where(iou_candidates > iou_threshold,
            is_in_topk, tensorlayerx.zeros_like(is_in_topk))
        is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes)
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask
        mask_positive_sum = mask_positive.sum(axis=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile([
                1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = tensorlayerx.where(mask_multiple_gts,
                is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        if self.force_gt_matching:
            is_max_iou = compute_max_iou_gt(ious) * pad_gt_mask
            mask_max_iou = (is_max_iou.sum(-2, keepdim=True) == 1).tile([1,
                num_max_boxes, 1])
            mask_positive = tensorlayerx.where(mask_max_iou, is_max_iou,
                mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        assigned_gt_index = mask_positive.argmax(axis=-2)
        batch_ind = tensorlayerx.ops.arange(start=0, dtype=gt_labels.dtype,
            limit=batch_size).unsqueeze(-1)
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = tensorlayerx.gather(gt_labels.flatten(),
            assigned_gt_index.flatten(), axis=0)
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        a0 = tensorlayerx.constant(x=assigned_labels, fill_value=bg_index,
            dtype=assigned_labels.dtype)
        b = mask_positive_sum > 0
        assigned_labels = tensorlayerx.where(mask_positive_sum > 0,
            assigned_labels, tensorlayerx.constant(x=assigned_labels,
            fill_value=bg_index, dtype=assigned_labels.dtype))
        assigned_bboxes = tensorlayerx.gather(gt_bboxes.reshape([-1, 4]),
            assigned_gt_index.flatten(), axis=0)
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])
        assigned_scores = tensorlayerx.ops.OneHot(assigned_labels, self.
            num_classes + 1)
        ind = list(range(self.num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = tensorlayerx.index_select(assigned_scores,
            tensorlayerx.convert_to_tensor(ind), axis=-1)
        if pred_bboxes is not None:
            ious = batch_iou_similarity(gt_bboxes, pred_bboxes) * mask_positive
            ious = ious.max(axis=-2).unsqueeze(-1)
            assigned_scores *= ious
        elif gt_scores is not None:
            gather_scores = tensorlayerx.gather(gt_scores.flatten(),
                assigned_gt_index.flatten(), axis=0)
            gather_scores = gather_scores.reshape([batch_size, num_anchors])
            gather_scores = tensorlayerx.where(mask_positive_sum > 0,
                gather_scores, tensorlayerx.zeros_like(gather_scores))
            assigned_scores *= gather_scores.unsqueeze(-1)
        return assigned_labels, assigned_bboxes, assigned_scores
