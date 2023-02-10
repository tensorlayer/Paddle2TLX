import tensorlayerx as tlx
import tensorlayerx.nn as nn
from .common  import  _Pad2dbase

__all__ = [
    'tlx_Pad2d'            # F.pad
]

class tlx_Pad2d(nn.Module):
    """
    The :class:`ZeroPad2d` class is a 2D padding layer for image [batch, height, width, channel].

    Parameters
    ----------
    padding : tuple of 2 tuples of 2 ints.
            - If tuple of 2 tuples of 2 ints, interpreted as ``((top_pad, bottom_pad), (left_pad, right_pad))``.
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tlx.nn.Input([10, 100, 100, 3], name='input')
    >>> pad2d = tlx.nn.ZeroPad2d(padding=((3, 3), (4, 4)))(net)
    >>> print(pad2d)
    >>> output shape : (10, 106, 108, 3)

    """

    def __init__(
        self,
        padding,
        mode="constant",
        value=0.0,
        data_format='channels_first',
        name=None
    ):
        super().__init__(name)
        self.padding = padding
        self.data_format = data_format
        self.value = value
        self.mode = mode
        tlx.logging.info("tlx_Pad2d   %s: padding: %s" % (self.name, str(self.padding)))

        if not isinstance(self.padding, (int, tuple, list)):
            raise AssertionError("Padding should be of type `int` or `tuple` or `list`")

        self.build()
        self._built = True

    def __repr__(self):
        s = '{classname}(padding={padding}'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        # self.layer = tlx.ops.ZeroPadding2D(padding=self.padding, data_format=self.data_format,mode=self.mode)
        self.layer = _Pad2dbase(padding=self.padding,value=self.value,data_format=self.data_format,mode=self.mode)

    def forward(self, inputs):
        # print(f"tlx_Pad2D.inputs.shape before={inputs.shape}")
        outputs = self.layer(inputs)
        # print(f"tlx_Pad2D.outputs.shape before={outputs.shape}")
        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        # print(f"tlx_Pad2D.outputs.shape after ={outputs.shape}")
        return outputs
