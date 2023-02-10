from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from core.workspace import register
__all__ = ['SmoothL1Loss']


@register
class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss.
    Args:
        beta (float): controls smooth region, it becomes L1 Loss when beta=0.0
        loss_weight (float): the final loss will be multiplied by this 
    """

    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        assert beta >= 0
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, pred, target, reduction='none'):
        """forward function, based on fvcore.
        Args:
            pred (Tensor): prediction tensor
            target (Tensor): target tensor, pred.shape must be the same as target.shape
            reduction (str): the way to reduce loss, one of (none, sum, mean)
        """
        assert reduction in ('none', 'sum', 'mean')
        target = target.detach()
        if self.beta < 1e-05:
            loss = tensorlayerx.ops.abs(pred - target)
        else:
            n = tensorlayerx.ops.abs(pred - target)
            cond = n < self.beta
            loss = tensorlayerx.where(cond, 0.5 * n ** 2 / self.beta, n - 
                0.5 * self.beta)
        if reduction == 'mean':
            loss = loss.mean() if loss.size > 0 else 0.0 * loss.sum()
        elif reduction == 'sum':
            loss = loss.sum()
        return loss * self.loss_weight
