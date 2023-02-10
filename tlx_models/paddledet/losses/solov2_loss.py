from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
from core.workspace import register
from core.workspace import serializable
__all__ = ['SOLOv2Loss']


@register
@serializable
class SOLOv2Loss(object):
    """
    SOLOv2Loss
    Args:
        ins_loss_weight (float): Weight of instance loss.
        focal_loss_gamma (float): Gamma parameter for focal loss.
        focal_loss_alpha (float): Alpha parameter for focal loss.
    """

    def __init__(self, ins_loss_weight=3.0, focal_loss_gamma=2.0,
        focal_loss_alpha=0.25):
        self.ins_loss_weight = ins_loss_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

    def _dice_loss(self, input, target):
        input = tensorlayerx.reshape(input, shape=(paddle2tlx.pd2tlx.ops.
            tlxops.tlx_get_tensor_shape(input)[0], -1))
        target = tensorlayerx.reshape(target, shape=(paddle2tlx.pd2tlx.ops.
            tlxops.tlx_get_tensor_shape(target)[0], -1))
        a = tensorlayerx.reduce_sum(input * target, axis=1)
        bb = tensorlayerx.reduce_sum(input * input, axis=1)
        b = bb + 0.001
        cc = tensorlayerx.reduce_sum(target * target, axis=1)
        c = cc + 0.001
        d = 2 * a / (b + c)
        return 1 - d

    def __call__(self, ins_pred_list, ins_label_list, cate_preds,
        cate_labels, num_ins):
        """
        Get loss of network of SOLOv2.
        Args:
            ins_pred_list (list): Variable list of instance branch output.
            ins_label_list (list): List of instance labels pre batch.
            cate_preds (list): Concat Variable list of categroy branch output.
            cate_labels (list): Concat list of categroy labels pre batch.
            num_ins (int): Number of positive samples in a mini-batch.
        Returns:
            loss_ins (Variable): The instance loss Variable of SOLOv2 network.
            loss_cate (Variable): The category loss Variable of SOLOv2 network.
        """
        loss_ins = []
        total_weights = tensorlayerx.zeros(shape=[1], dtype='float32')
        for input, target in zip(ins_pred_list, ins_label_list):
            if input is None:
                continue
            target = tensorlayerx.cast(target, 'float32')
            target = tensorlayerx.reshape(target, shape=[-1, paddle2tlx.
                pd2tlx.ops.tlxops.tlx_get_tensor_shape(input)[-2],
                paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(input)[-1]])
            weights = tensorlayerx.cast(tensorlayerx.reduce_sum(target,
                axis=[1, 2]) > 0, 'float32')
            input = tensorlayerx.ops.sigmoid(input)
            dice_out = tensorlayerx.ops.multiply(self._dice_loss(input,
                target), weights)
            total_weights += tensorlayerx.reduce_sum(weights)
            loss_ins.append(dice_out)
        aa = tensorlayerx.concat(loss_ins)
        bb = tensorlayerx.reduce_sum(aa)
        loss_ins = bb / total_weights
        loss_ins = loss_ins * self.ins_loss_weight
        num_classes = cate_preds.shape[-1]
        cate_labels_bin = tensorlayerx.ops.OneHot(cate_labels, num_classes=\
            num_classes + 1)
        cate_labels_bin = cate_labels_bin[:, 1:]
        loss_cate = paddle.nn.functional.sigmoid_focal_loss(cate_preds,
            label=cate_labels_bin, normalizer=num_ins + 1.0, gamma=self.
            focal_loss_gamma, alpha=self.focal_loss_alpha)
        return loss_ins, loss_cate
