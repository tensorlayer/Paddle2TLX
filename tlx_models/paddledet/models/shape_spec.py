import tensorlayerx as tlx
import paddle
import paddle2tlx
from collections import namedtuple


class ShapeSpec(namedtuple('_ShapeSpec', ['channels', 'height', 'width',
    'stride'])):

    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super(ShapeSpec, cls).__new__(cls, channels, height, width,
            stride)
