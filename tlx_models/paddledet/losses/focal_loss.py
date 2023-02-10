from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from core.workspace import register
__all__ = ['FocalLoss']


@register
class FocalLoss(nn.Module):
    """A wrapper around paddle.nn.functional.sigmoid_focal_loss.
    Args:
        use_sigmoid (bool): currently only support use_sigmoid=True
        alpha (float): parameter alpha in Focal Loss
        gamma (float): parameter gamma in Focal Loss
        loss_weight (float): final loss will be multiplied by this
    """

    def __init__(self, use_sigmoid=True, alpha=0.25, gamma=2.0, loss_weight=1.0
        ):
        super(FocalLoss, self).__init__()
        assert use_sigmoid == True, 'Focal Loss only supports sigmoid at the moment'
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, pred, target, reduction='none'):
        """forward function.
        Args:
            pred (Tensor): logits of class prediction, of shape (N, num_classes)
            target (Tensor): target class label, of shape (N, )
            reduction (str): the way to reduce loss, one of (none, sum, mean)
        """
        num_classes = pred.shape[1]
        target = tensorlayerx.ops.OneHot(target, num_classes + 1).cast(pred
            .dtype)
        target = target[:, :-1].detach()
        loss = paddle.nn.functional.sigmoid_focal_loss(pred, target, alpha=\
            self.alpha, gamma=self.gamma, reduction=reduction)
        return loss * self.loss_weight
