
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import paddle 
from .download import get_weights_path

from .logger import setup_logger
from paddle2tlx.pd2tlx.utils import load_model_det
logger = setup_logger(__name__)


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://') \
            or path.startswith('https://') \
            or path.startswith('det://')


def _strip_postfix(path):
    path, ext = os.path.splitext(path)
    assert ext in ['', '.pdparams', '.pdopt', '.pdmodel'], \
            "Unknown postfix {} from weights".format(ext)
    return path


# def match_state_dict(model_state_dict, weight_state_dict):
#     """
#     Match between the model state dict and pretrained weight state dict.
#     Return the matched state dict.
#
#     The method supposes that all the names in pretrained weight state dict are
#     subclass of the names in models`, if the prefix 'backbone.' in pretrained weight
#     keys is stripped. And we could get the candidates for each model key. Then we
#     select the name with the longest matched size as the final match result. For
#     example, the model state dict has the name of
#     'backbone.res2.res2a.branch2a.conv.weight' and the pretrained weight as
#     name of 'res2.res2a.branch2a.conv.weight' and 'branch2a.conv.weight'. We
#     match the 'res2.res2a.branch2a.conv.weight' to the model key.
#     """
#
#     model_keys = sorted(model_state_dict.keys())
#     weight_keys = sorted(weight_state_dict.keys())
#
#     def match(a, b):
#         if b.startswith('backbone.res5'):
#             # In Faster RCNN, res5 pretrained weights have prefix of backbone,
#             # however, the corresponding model weights have difficult prefix,
#             # bbox_head.
#             b = b[9:]
#         return a == b or a.endswith("." + b)
#
#     match_matrix = np.zeros([len(model_keys), len(weight_keys)])
#     for i, m_k in enumerate(model_keys):
#         for j, w_k in enumerate(weight_keys):
#             if match(m_k, w_k):
#                 match_matrix[i, j] = len(w_k)
#     max_id = match_matrix.argmax(1)
#     max_len = match_matrix.max(1)
#     max_id[max_len == 0] = -1
#
#     load_id = set(max_id)
#     load_id.discard(-1)
#     not_load_weight_name = []
#     for idx in range(len(weight_keys)):
#         if idx not in load_id:
#             not_load_weight_name.append(weight_keys[idx])
#
#     if len(not_load_weight_name) > 0:
#         logger.info('{} in pretrained weight is not used in the model, '
#                     'and its will not be loaded'.format(not_load_weight_name))
#     matched_keys = {}
#     result_state_dict = {}
#     for model_id, weight_id in enumerate(max_id):
#         if weight_id == -1:
#             continue
#         model_key = model_keys[model_id]
#         weight_key = weight_keys[weight_id]
#         weight_value = weight_state_dict[weight_key]
#         model_value_shape = list(model_state_dict[model_key].shape)
#
#         if list(weight_value.shape) != model_value_shape:
#             logger.info(
#                 'The shape {} in pretrained weight {} is unmatched with '
#                 'the shape {} in model {}. And the weight {} will not be '
#                 'loaded'.format(weight_value.shape, weight_key,
#                                 model_value_shape, model_key, weight_key))
#             continue
#
#         assert model_key not in result_state_dict
#         result_state_dict[model_key] = weight_value
#         if weight_key in matched_keys:
#             raise ValueError('Ambiguity weight {} loaded, it matches at least '
#                              '{} and {} in the model'.format(
#                                  weight_key, model_key, matched_keys[
#                                      weight_key]))
#         matched_keys[weight_key] = model_key
#     return result_state_dict



def load_pretrain_weight(model, pretrain_weight):
    if is_url(pretrain_weight):
        pretrain_weight = get_weights_path(pretrain_weight)

    # path = _strip_postfix(pretrain_weight)
    # if not (os.path.isdir(path) or os.path.isfile(path) or
    #         os.path.exists(path + '.pdparams')):
    #     raise ValueError("Model pretrain path `{}` does not exists. "
    #                      "If you don't want to load pretrain model, "
    #                      "please delete `pretrain_weights` field in "
    #                      "config file.".format(path))
    # weights_path = path + '.pdparams'
    _ = load_model_det(model, pretrain_weight)
    # model_dict = model.state_dict()
    # param_state_dict = paddle.load(weights_path)
    # param_state_dict = match_state_dict(model_dict, param_state_dict)
    #
    # for k, v in param_state_dict.items():
    #     if isinstance(v, np.ndarray):
    #         v = paddle.to_tensor(v)
    #     if model_dict[k].dtype != v.dtype:
    #         param_state_dict[k] = v.astype(model_dict[k].dtype)
    #
    # model.set_dict(param_state_dict)
    # logger.info('Finish loading model weights: {}'.format(weights_path))
