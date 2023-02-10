from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import tensorlayerx as tlx
import paddle
import paddle2tlx
import os
import numpy as np
import tensorlayerx
from .download import get_weights_path
from .logger import setup_logger
from paddle2tlx.pd2tlx.utils import restore_model_det
logger = setup_logger(__name__)


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://') or path.startswith('https://'
        ) or path.startswith('det://')


def _strip_postfix(path):
    path, ext = os.path.splitext(path)
    assert ext in ['', '.pdparams', '.pdopt', '.pdmodel'
        ], 'Unknown postfix {} from weights'.format(ext)
    return path


def load_pretrain_weight(model, pretrain_weight):
    if is_url(pretrain_weight):
        pretrain_weight = get_weights_path(pretrain_weight)
    _ = restore_model_det(model, pretrain_weight)
