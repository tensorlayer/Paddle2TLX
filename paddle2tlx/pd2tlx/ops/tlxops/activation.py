import tensorlayerx as tlx
import tensorlayerx.nn as nn

__all__ = ["tlx_GELU"]

class tlx_GELU(nn.Module):
    r"""
    GELU Activation.
    If approximate is True
    .. math::
        GELU(x) = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))
    else
    .. math::
        GELU(x) = 0.5 * x * (1 + erf(\frac{x}{\sqrt{2}}))
    Parameters:
        approximate (bool, optional): Wether to enable approximation. Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.
    Examples:
        .. code-block:: python
            import paddle
            import numpy as np
            x = paddle.to_tensor(np.array([[-1, 0.5],[1, 1.5]]))
            m = paddle.nn.GELU()
            out = m(x) # [-0.158655 0.345731 0.841345 1.39979]
            m = paddle.nn.GELU(True)
            out = m(x) # [-0.158808 0.345714 0.841192 1.39957]
    """

    def __init__(self, approximate=False, name=None):
        super(tlx_GELU, self).__init__()
        self._approximate = approximate
        self._name = name

    def forward(self, x):
        return tlx.ops.gelu(x, self._approximate) #, self._name)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'approximate={}{}'.format(self._approximate, name_str)