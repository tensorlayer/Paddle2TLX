from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
from core.workspace import register
from core.workspace import serializable
from .iou_loss import IouLoss
from models.bbox_utils import bbox_iou


@register
@serializable
class IouAwareLoss(IouLoss):
    """
    iou aware loss, see https://arxiv.org/abs/1912.05992
    Args:
        loss_weight (float): iou aware loss weight, default is 1.0
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
    """

    def __init__(self, loss_weight=1.0, giou=False, diou=False, ciou=False):
        super(IouAwareLoss, self).__init__(loss_weight=loss_weight, giou=\
            giou, diou=diou, ciou=ciou)

    def __call__(self, ioup, pbox, gbox):
        iou = bbox_iou(pbox, gbox, giou=self.giou, diou=self.diou, ciou=\
            self.ciou)
        iou.stop_gradient = True
        loss_iou_aware = tensorlayerx.losses.sigmoid_cross_entropy(ioup,
            iou, reduction='none')
        loss_iou_aware = loss_iou_aware * self.loss_weight
        return loss_iou_aware
