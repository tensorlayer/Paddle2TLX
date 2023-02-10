import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import numpy as np
from losses.varifocal_loss import varifocal_loss
from models.bbox_utils import batch_bbox_overlaps
from core.workspace import register


@register
class SimOTAAssigner(object):
    """Computes matching between predictions and ground truth.
    Args:
        center_radius (int | float, optional): Ground truth center size
            to judge whether a prior is in center. Default 2.5.
        candidate_topk (int, optional): The candidate top-k which used to
            get top-k ious to calculate dynamic-k. Default 10.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 3.0.
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        num_classes (int): The num_classes of dataset.
        use_vfl (int): Whether to use varifocal_loss when calculating the cost matrix.
    """
    __shared__ = ['num_classes']

    def __init__(self, center_radius=2.5, candidate_topk=10, iou_weight=3.0,
        cls_weight=1.0, num_classes=80, use_vfl=True):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.num_classes = num_classes
        self.use_vfl = use_vfl

    def get_in_gt_and_in_center_info(self, flatten_center_and_stride, gt_bboxes
        ):
        num_gt = gt_bboxes.shape[0]
        flatten_x = flatten_center_and_stride[:, 0].unsqueeze(1).tile([1,
            num_gt])
        flatten_y = flatten_center_and_stride[:, 1].unsqueeze(1).tile([1,
            num_gt])
        flatten_stride_x = flatten_center_and_stride[:, 2].unsqueeze(1).tile([
            1, num_gt])
        flatten_stride_y = flatten_center_and_stride[:, 3].unsqueeze(1).tile([
            1, num_gt])
        l_ = flatten_x - gt_bboxes[:, (0)]
        t_ = flatten_y - gt_bboxes[:, (1)]
        r_ = gt_bboxes[:, (2)] - flatten_x
        b_ = gt_bboxes[:, (3)] - flatten_y
        deltas = tensorlayerx.ops.stack([l_, t_, r_, b_], axis=1)
        is_in_gts = deltas.min(axis=1) > 0
        is_in_gts_all = is_in_gts.sum(axis=1) > 0
        gt_center_xs = (gt_bboxes[:, (0)] + gt_bboxes[:, (2)]) / 2.0
        gt_center_ys = (gt_bboxes[:, (1)] + gt_bboxes[:, (3)]) / 2.0
        ct_bound_l = gt_center_xs - self.center_radius * flatten_stride_x
        ct_bound_t = gt_center_ys - self.center_radius * flatten_stride_y
        ct_bound_r = gt_center_xs + self.center_radius * flatten_stride_x
        ct_bound_b = gt_center_ys + self.center_radius * flatten_stride_y
        cl_ = flatten_x - ct_bound_l
        ct_ = flatten_y - ct_bound_t
        cr_ = ct_bound_r - flatten_x
        cb_ = ct_bound_b - flatten_y
        ct_deltas = tensorlayerx.ops.stack([cl_, ct_, cr_, cb_], axis=1)
        is_in_cts = ct_deltas.min(axis=1) > 0
        is_in_cts_all = is_in_cts.sum(axis=1) > 0
        is_in_gts_or_centers_all = tensorlayerx.ops.logical_or(is_in_gts_all,
            is_in_cts_all)
        is_in_gts_or_centers_all_inds = (paddle2tlx.pd2tlx.ops.tlxops.
            tlx_nonzero(is_in_gts_or_centers_all).squeeze(1))
        is_in_gts_and_centers = tensorlayerx.ops.logical_and(tensorlayerx.
            gather(is_in_gts.cast('int'), is_in_gts_or_centers_all_inds,
            axis=0).cast('bool'), tensorlayerx.gather(is_in_cts.cast('int'),
            is_in_gts_or_centers_all_inds, axis=0).cast('bool'))
        return (is_in_gts_or_centers_all, is_in_gts_or_centers_all_inds,
            is_in_gts_and_centers)

    def dynamic_k_matching(self, cost_matrix, pairwise_ious, num_gt):
        match_matrix = np.zeros_like(cost_matrix.numpy())
        topk_ious, _ = tensorlayerx.ops.topk(pairwise_ious, min(self.
            candidate_topk, pairwise_ious.shape[0]), axis=0)
        dynamic_ks = tensorlayerx.ops.clip_by_value(topk_ious.sum(0).cast(
            'int'), clip_value_min=1, clip_value_max=None)
        for gt_idx in range(num_gt):
            _, pos_idx = tensorlayerx.ops.topk(cost_matrix[:, gt_idx], k=\
                dynamic_ks[gt_idx], largest=False)
            match_matrix[:, gt_idx][pos_idx.numpy()] = 1.0
        del topk_ious, dynamic_ks, pos_idx
        extra_match_gts_mask = match_matrix.sum(1) > 1
        if extra_match_gts_mask.sum() > 0:
            cost_matrix = cost_matrix.numpy()
            cost_argmin = np.argmin(cost_matrix[extra_match_gts_mask, :],
                axis=1)
            match_matrix[extra_match_gts_mask, :] *= 0.0
            match_matrix[extra_match_gts_mask, cost_argmin] = 1.0
        match_fg_mask_inmatrix = match_matrix.sum(1) > 0
        match_gt_inds_to_fg = match_matrix[match_fg_mask_inmatrix, :].argmax(1)
        return match_gt_inds_to_fg, match_fg_mask_inmatrix

    def get_sample(self, assign_gt_inds, gt_bboxes):
        pos_inds = np.unique(np.nonzero(assign_gt_inds > 0)[0])
        neg_inds = np.unique(np.nonzero(assign_gt_inds == 0)[0])
        pos_assigned_gt_inds = assign_gt_inds[pos_inds] - 1
        if gt_bboxes.size == 0:
            assert pos_assigned_gt_inds.size == 0
            pos_gt_bboxes = np.empty_like(gt_bboxes).reshape(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.resize(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def __call__(self, flatten_cls_pred_scores, flatten_center_and_stride,
        flatten_bboxes, gt_bboxes, gt_labels, eps=1e-07):
        """Assign gt to priors using SimOTA.
        TODO: add comment.
        Returns:
            assign_result: The assigned result.
        """
        num_gt = gt_bboxes.shape[0]
        num_bboxes = flatten_bboxes.shape[0]
        if num_gt == 0 or num_bboxes == 0:
            label = np.ones([num_bboxes], dtype=np.int64) * self.num_classes
            label_weight = np.ones([num_bboxes], dtype=np.float32)
            bbox_target = np.zeros_like(flatten_center_and_stride)
            return 0, label, label_weight, bbox_target
        (is_in_gts_or_centers_all, is_in_gts_or_centers_all_inds,
            is_in_boxes_and_center) = (self.get_in_gt_and_in_center_info(
            flatten_center_and_stride, gt_bboxes))
        valid_flatten_bboxes = flatten_bboxes[is_in_gts_or_centers_all_inds]
        valid_cls_pred_scores = flatten_cls_pred_scores[
            is_in_gts_or_centers_all_inds]
        num_valid_bboxes = valid_flatten_bboxes.shape[0]
        pairwise_ious = batch_bbox_overlaps(valid_flatten_bboxes, gt_bboxes)
        if self.use_vfl:
            gt_vfl_labels = gt_labels.squeeze(-1).unsqueeze(0).tile([
                num_valid_bboxes, 1]).reshape([-1])
            valid_pred_scores = valid_cls_pred_scores.unsqueeze(1).tile([1,
                num_gt, 1]).reshape([-1, self.num_classes])
            vfl_score = np.zeros(valid_pred_scores.shape)
            vfl_score[np.arange(0, vfl_score.shape[0]), gt_vfl_labels.numpy()
                ] = pairwise_ious.reshape([-1])
            vfl_score = tensorlayerx.convert_to_tensor(vfl_score)
            losses_vfl = varifocal_loss(valid_pred_scores, vfl_score,
                use_sigmoid=False).reshape([num_valid_bboxes, num_gt])
            losses_giou = batch_bbox_overlaps(valid_flatten_bboxes,
                gt_bboxes, mode='giou')
            aa = tensorlayerx.ops.logical_not(is_in_boxes_and_center).cast(
                'float32')
            cost_matrix = (losses_vfl * self.cls_weight + losses_giou *
                self.iou_weight + aa * 100000000)
        else:
            iou_cost = -tensorlayerx.ops.log(pairwise_ious + eps)
            gt_onehot_label = tensorlayerx.ops.OneHot(gt_labels.squeeze(-1)
                .cast('int64'), flatten_cls_pred_scores.shape[-1]).cast(
                'float32').unsqueeze(0).tile([num_valid_bboxes, 1, 1])
            valid_pred_scores = valid_cls_pred_scores.unsqueeze(1).tile([1,
                num_gt, 1])
            cls_cost = tensorlayerx.losses.binary_cross_entropy(
                valid_pred_scores, gt_onehot_label, reduction='none').sum(-1)
            aa = tensorlayerx.ops.logical_not(is_in_boxes_and_center).cast(
                'float32')
            cost_matrix = (cls_cost * self.cls_weight + iou_cost * self.
                iou_weight + aa * 100000000)
        match_gt_inds_to_fg, match_fg_mask_inmatrix = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt)
        assigned_gt_inds = np.zeros([num_bboxes], dtype=np.int64)
        match_fg_mask_inall = np.zeros_like(assigned_gt_inds)
        match_fg_mask_inall[is_in_gts_or_centers_all.numpy()
            ] = match_fg_mask_inmatrix
        assigned_gt_inds[match_fg_mask_inall.astype(np.bool)
            ] = match_gt_inds_to_fg + 1
        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = (self.
            get_sample(assigned_gt_inds, gt_bboxes.numpy()))
        bbox_target = np.zeros_like(flatten_bboxes)
        bbox_weight = np.zeros_like(flatten_bboxes)
        label = np.ones([num_bboxes], dtype=np.int64) * self.num_classes
        label_weight = np.zeros([num_bboxes], dtype=np.float32)
        if len(pos_inds) > 0:
            gt_labels = gt_labels.numpy()
            pos_bbox_targets = pos_gt_bboxes
            bbox_target[pos_inds, :] = pos_bbox_targets
            bbox_weight[pos_inds, :] = 1.0
            if not np.any(gt_labels):
                label[pos_inds] = 0
            else:
                label[pos_inds] = gt_labels.squeeze(-1)[pos_assigned_gt_inds]
            label_weight[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weight[neg_inds] = 1.0
        pos_num = max(pos_inds.size, 1)
        return pos_num, label, label_weight, bbox_target
