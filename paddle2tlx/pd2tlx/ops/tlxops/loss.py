
import tensorlayerx  as tlx
import tensorlayerx.nn as nn
import paddle.nn.functional as F
from tensorlayerx.backend.ops.paddle_nn import _C_ops


__all__ = [
    'tlx_L1Loss',       # ok
    'tlx_BCEWithLogitsLoss',    # F.binary_cross_entropy_with_logits
    'tlx_MSELoss',
    "tlx_cross_entropy"   # _C_ops.softmax_with_cross_entropy
]

class tlx_L1Loss(nn.Module):
    def __init__(self, reduction='mean', name=None):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in L1Loss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction
            )
        super(tlx_L1Loss, self).__init__()
        self.reduction = reduction
        self.name = name

    def forward(self, input, label):
        return tlx.losses.absolute_difference_error(
            output=input, target=label, reduction=self.reduction
        )

class tlx_MSELoss(nn.Module):
    def __init__(self,
                 reduction='mean',):
        super(tlx_MSELoss, self).__init__()
        self.reduction = reduction

    def __repr__(self):
        if self.reduction not in ['sum', 'mean', 'none']:
           s =  f"""The value of 'reduction' in BCEWithLogitsLoss should be 'sum', 'mean' or 'none', but "
                "received {self.reduction}, which is not allowed."""
        return s

    def forward(self, input, label):
        out = tlx.losses.mean_squared_error(output=input, target=label, reduction=self.reduction)
        return out


class tlx_BCEWithLogitsLoss(nn.Module):
    r"""
    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 50, 50, 32], name='input')
    >>> net = tlx.nn.BatchNorm()(net)

    Notes
    -----
    The :class:`BatchNorm` is universally suitable for 3D/4D/5D input in static model, but should not be used
    in dynamic model where layer is built upon class initialization. So the argument 'num_features' should only be used
    for subclasses :class:`BatchNorm1d`, :class:`BatchNorm2d` and :class:`BatchNorm3d`. All the three subclasses are
    suitable under all kinds of conditions.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`__
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`__

    """
    def __init__(self,
                 weight=None,
                 reduction='mean',
                 pos_weight=None,
                 name=None):
        super(tlx_BCEWithLogitsLoss, self).__init__(name=name)
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.name = name

    def __repr__(self):
        if self.reduction not in ['sum', 'mean', 'none']:
           s =  f"""The value of 'reduction' in BCEWithLogitsLoss should be 'sum', 'mean' or 'none', but "
                "received {self.reduction}, which is not allowed."""
        return s

    def forward(self, logit, label):
        out = F.binary_cross_entropy_with_logits(
            logit, label, self.weight, self.reduction, self.pos_weight,
            self.name)
        return out


def tlx_cross_entropy(input, label, weight=None, ignore_index=-100,
    reduction='mean', soft_label=False, axis=-1, use_softmax=True, name=None):
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in softmax_cross_entropyshould be 'sum', 'mean' or 'none', but received %s, which is not allowed."
             % reduction)
    if ignore_index > 0 and soft_label == True:
        raise ValueError(
            "When soft_label == True, the value of 'ignore_index' in softmax_cross_entropyshould be '-100', but received %s, which is not allowed."
             % ignore_index)
    input_dims = len(list(input.shape))
    if input_dims == 0:
        raise ValueError('The dimention of input should be larger than zero!')
    label_dims = len(list(label.shape))
    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            'Expected nput_dims - 1 = label_dims or input_dims == label_dims (got nput_dims{}, label_dims{})'
            .format(input_dims, label_dims))
    if input_dims - 1 == label_dims:
        label = tlx.ops.expand_dims(label, axis=axis)
    if soft_label == False:
        valid_label = tlx.cast(label != ignore_index, dtype=label.dtype
            ) * label
        label_min = tlx.reduce_min(valid_label)
        label_max = tlx.reduce_max(valid_label)
        if label_min < 0:
            raise ValueError('Target {} is out of lower bound.'.format(
                label_min.item()))
        if label_max >= input.shape[axis]:
            raise ValueError('Target {} is out of upper bound.'.format(
                label_max.item()))
    _, out = _C_ops.softmax_with_cross_entropy(input, label, 'soft_label',
        soft_label, 'ignore_index', ignore_index, 'numeric_stable_mode',
        True, 'axis', axis, 'use_softmax', use_softmax)
    if input_dims - 1 == label_dims:
        out = tlx.ops.squeeze(out, axis=axis)
    return out
