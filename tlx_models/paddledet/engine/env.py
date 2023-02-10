from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import random
import numpy as np
import tensorlayerx
__all__ = ['set_random_seed']


def set_random_seed(seed):
    tensorlayerx.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
