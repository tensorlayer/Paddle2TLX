from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
from core.workspace import register
from models.proposal_generator.target import label_box
__all__ = ['MaxIoUAssigner']


@register
class MaxIoUAssigner(object):
    """a standard bbox assigner based on max IoU, use det's label_box 
    as backend.
    Args:
        positive_overlap (float): threshold for defining positive samples 
        negative_overlap (float): threshold for denining negative samples
        allow_low_quality (bool): whether to lower IoU thr if a GT poorly
            overlaps with candidate bboxes
    """

    def __init__(self, positive_overlap, negative_overlap,
        allow_low_quality=True):
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap
        self.allow_low_quality = allow_low_quality

    def __call__(self, bboxes, gt_bboxes):
        matches, match_labels = label_box(bboxes, gt_bboxes,
            positive_overlap=self.positive_overlap, negative_overlap=self.
            negative_overlap, allow_low_quality=self.allow_low_quality,
            ignore_thresh=-1, is_crowd=None, assign_on_cpu=False)
        return matches, match_labels
