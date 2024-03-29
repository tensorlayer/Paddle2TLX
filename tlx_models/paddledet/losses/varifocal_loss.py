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
from core.workspace import serializable
from models import ops
__all__ = ['VarifocalLoss']


def varifocal_loss(pred, target, alpha=0.75, gamma=2.0, iou_weighted=True,
    use_sigmoid=True):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

    Args:
        pred (Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
    """
    assert pred.shape == target.shape
    if use_sigmoid:
        pred_new = tensorlayerx.ops.sigmoid(pred)
    else:
        pred_new = pred
    target = target.cast(pred.dtype)
    if iou_weighted:
        focal_weight = target * (target > 0.0).cast('float32') + alpha * (
            pred_new - target).abs().pow(gamma) * (target <= 0.0).cast(
            'float32')
    else:
        focal_weight = (target > 0.0).cast('float32') + alpha * (pred_new -
            target).abs().pow(gamma) * (target <= 0.0).cast('float32')
    if use_sigmoid:
        loss = tensorlayerx.losses.sigmoid_cross_entropy(pred, target,
            reduction='none')
        loss = loss * focal_weight
    else:
        loss = tensorlayerx.losses.binary_cross_entropy(pred, target,
            reduction='none')
        loss = loss * focal_weight
        loss = loss.sum(axis=1)
    return loss


@register
@serializable
class VarifocalLoss(nn.Module):

    def __init__(self, use_sigmoid=True, alpha=0.75, gamma=2.0,
        iou_weighted=True, reduction='mean', loss_weight=1.0):
        """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

        Args:
            use_sigmoid (bool, optional): Whether the prediction is
                used for sigmoid or softmax. Defaults to True.
            alpha (float, optional): A balance factor for the negative part of
                Varifocal Loss, which is different from the alpha of Focal
                Loss. Defaults to 0.75.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            iou_weighted (bool, optional): Whether to weight the loss of the
                positive examples with the iou target. Defaults to True.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(VarifocalLoss, self).__init__()
        assert alpha >= 0.0
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        Returns:
            Tensor: The calculated loss
        """
        loss = self.loss_weight * varifocal_loss(pred, target, alpha=self.
            alpha, gamma=self.gamma, iou_weighted=self.iou_weighted,
            use_sigmoid=self.use_sigmoid)
        if weight is not None:
            loss = loss * weight
        if avg_factor is None:
            if self.reduction == 'none':
                return loss
            elif self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
        elif self.reduction == 'mean':
            loss = loss.sum() / avg_factor
        elif self.reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
        return loss
