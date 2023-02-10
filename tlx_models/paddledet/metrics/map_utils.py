from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import tensorlayerx as tlx
import paddle
import paddle2tlx
from utils.logger import setup_logger
logger = setup_logger(__name__)
__all__ = ['bbox_area']


def bbox_area(bbox, is_bbox_normalized):
    """
    Calculate area of a bounding box
    """
    norm = 1.0 - float(is_bbox_normalized)
    width = bbox[2] - bbox[0] + norm
    height = bbox[3] - bbox[1] + norm
    return width * height
