from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
from core.workspace import register
from core.workspace import serializable
__all__ = ['CTFocalLoss']


@register
@serializable
class CTFocalLoss(object):
    """
    CTFocalLoss: CornerNet & CenterNet Focal Loss
    Args:
        loss_weight (float): loss weight
        gamma (float): gamma parameter for Focal Loss
    """

    def __init__(self, loss_weight=1.0, gamma=2.0):
        self.loss_weight = loss_weight
        self.gamma = gamma

    def __call__(self, pred, target):
        """
        Calculate the loss
        Args:
            pred (Tensor): heatmap prediction
            target (Tensor): target for positive samples
        Return:
            ct_focal_loss (Tensor): Focal Loss used in CornerNet & CenterNet.
                Note that the values in target are in [0, 1] since gaussian is
                used to reduce the punishment and we treat [0, 1) as neg example.
        """
        fg_map = tensorlayerx.cast(target == 1, 'float32')
        fg_map.stop_gradient = True
        bg_map = tensorlayerx.cast(target < 1, 'float32')
        bg_map.stop_gradient = True
        neg_weights = tensorlayerx.pow(1 - target, 4)
        aa = tensorlayerx.ops.log(pred)
        bb = tensorlayerx.ops.log(1 - pred)
        cc = tensorlayerx.pow(1 - pred, self.gamma)
        dd = tensorlayerx.pow(pred, self.gamma)
        pos_loss = 0 - aa * cc * fg_map
        neg_loss = 0 - bb * dd * neg_weights * bg_map
        pos_loss = tensorlayerx.reduce_sum(pos_loss)
        neg_loss = tensorlayerx.reduce_sum(neg_loss)
        fg_num = tensorlayerx.reduce_sum(fg_map)
        aa = tensorlayerx.cast(fg_num == 0, 'float32')
        ct_focal_loss = (pos_loss + neg_loss) / (fg_num + aa)
        return ct_focal_loss * self.loss_weight
