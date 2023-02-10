import os
import paddle
import pickle
import six
import numpy as np
import logging
from .load_model import get_path_from_url, tlx_load

logger = logging.getLogger(__name__)


def match_state_dict(model_state_dict, weight_state_dict):
    """
    Match between the model state dict and pretrained weight state dict.
    Return the matched state dict.

    The method supposes that all the names in pretrained weight state dict are
    subclass of the names in models`, if the prefix 'backbone.' in pretrained weight
    keys is stripped. And we could get the candidates for each model key. Then we
    select the name with the longest matched size as the final match result. For
    example, the model state dict has the name of
    'backbone.res2.res2a.branch2a.conv.weight' and the pretrained weight as
    name of 'res2.res2a.branch2a.conv.weight' and 'branch2a.conv.weight'. We
    match the 'res2.res2a.branch2a.conv.weight' to the model key.
    """

    model_keys = sorted(model_state_dict.keys())
    weight_keys = sorted(weight_state_dict.keys())

    def match(a, b):
        if b.startswith('backbone.res5'):
            # In Faster RCNN, res5 pretrained weights have prefix of backbone,
            # however, the corresponding model weights have difficult prefix,
            # bbox_head.
            b = b[9:]
        return a == b or a.endswith("." + b)

    match_matrix = np.zeros([len(model_keys), len(weight_keys)])
    for i, m_k in enumerate(model_keys):
        for j, w_k in enumerate(weight_keys):
            if match(m_k, w_k):
                match_matrix[i, j] = len(w_k)
    max_id = match_matrix.argmax(1)
    max_len = match_matrix.max(1)
    max_id[max_len == 0] = -1

    load_id = set(max_id)
    load_id.discard(-1)
    not_load_weight_name = []
    for idx in range(len(weight_keys)):
        if idx not in load_id:
            not_load_weight_name.append(weight_keys[idx])

    if len(not_load_weight_name) > 0:
        logger.info('{} in pretrained weight is not used in the model, '
                    'and its will not be loaded'.format(not_load_weight_name))
    matched_keys = {}
    result_state_dict = {}
    for model_id, weight_id in enumerate(max_id):
        if weight_id == -1:
            continue
        model_key = model_keys[model_id]
        weight_key = weight_keys[weight_id]
        weight_value = weight_state_dict[weight_key]
        model_value_shape = list(model_state_dict[model_key].shape)

        if list(weight_value.shape) != model_value_shape:
            logger.info(
                'The shape {} in pretrained weight {} is unmatched with '
                'the shape {} in model {}. And the weight {} will not be '
                'loaded'.format(weight_value.shape, weight_key,
                                model_value_shape, model_key, weight_key))
            continue

        assert model_key not in result_state_dict
        result_state_dict[model_key] = weight_value
        if weight_key in matched_keys:
            raise ValueError('Ambiguity weight {} loaded, it matches at least '
                             '{} and {} in the model'.format(
                weight_key, model_key, matched_keys[
                    weight_key]))
        matched_keys[weight_key] = model_key
    return result_state_dict


def get_new_key(param):
    new_param = dict()
    for name in param.keys():
        name_l = ".".join(name.split('.')[:-2])
        if '_batch_norm.weight' in name:
            new_key = name_l + "." + "my_batch_norm.weight"
            new_param[new_key] = param[name]
        elif '_batch_norm.bias' in name:
            new_key = name_l + "." + "my_batch_norm.bias"
            new_param[new_key] = param[name]
        elif '_batch_norm._mean' in name:
            new_key = name_l + "." + "my_batch_norm._mean"
            new_param[new_key] = param[name]
        elif '_batch_norm._variance' in name:
            new_key = name_l + "." + "my_batch_norm._variance"
            new_param[new_key] = param[name]
        else:
            new_param[name] = param[name]
    return new_param


def load_model_det(model, pretrained_model_file):
    param_state_dict = paddle.load(pretrained_model_file)
    param_state_dict = get_new_key(param_state_dict)
    model_dict = model.state_dict()

    # [print(k, np.shape(v)) for k,v in model.state_dict().items()]
    param_state_dict = match_state_dict(model_dict, param_state_dict)

    for k, v in param_state_dict.items():
        if isinstance(v, np.ndarray):
            v = paddle.to_tensor(v)
        if model_dict[k].dtype != v.dtype:
            param_state_dict[k] = v.astype(model_dict[k].dtype)

    model.set_dict(param_state_dict)
    logger.info('Finish loading model weights: {}'.format(pretrained_model_file))
    return model


def restore_model_det(model, pretrained_model_file):
    param_state_dict = tlx_load(pretrained_model_file)
    restore_model(param_state_dict, model)
    logger.info('Finish loading model weights: {}'.format(pretrained_model_file))
    return model


def restore_model(param, model):
    from tensorlayerx.files import assign_weights
    tlx2pd_namelast = {'filters': 'weight',
                       'gamma': 'weight',
                       'weights': 'weight',
                       'embeddings': 'weight',
                       'q_weight': 'q_proj.weight',
                       'k_weight': 'k_proj.weight',
                       'v_weight': 'v_proj.weight',
                       'out_weight': 'out_proj.weight',
                       'q_bias': 'q_proj.bias',
                       'k_bias': 'k_proj.bias',
                       'v_bias': 'v_proj.bias',
                       'out_bias': 'out_proj.bias',
                       'beta': 'bias',
                       'biases': 'bias',
                       'moving_mean': '_mean',
                       'moving_var': '_variance',
                       'my_batch_norm.gamma': '_batch_norm.weight',
                       'my_batch_norm.beta': '_batch_norm.bias',
                       'my_batch_norm.moving_mean': '_batch_norm._mean',
                       'my_batch_norm.moving_var': '_batch_norm._variance', }

    model_state = [i for i, k in model.named_parameters()]
    weights = []
    # get_param_info(model)

    for i in range(len(model_state)):
        model_key = model_state[i]
        model_keys = model_key.rsplit('.', 1)
        indices = [i for i, c in enumerate(model_key) if c == '.']
        model_key_l, model_key_r = '', ''
        if len(indices) >= 2:
            model_key_l = model_key[:indices[-2]]
            model_key_r = model_key[indices[-2] + 1:]
        if len(model_keys) == 2:
            if model_keys[1] in tlx2pd_namelast and model_key_r not in tlx2pd_namelast:
                model_key = model_keys[0] + '.' + tlx2pd_namelast[model_keys[1]]
            elif model_keys[1] in tlx2pd_namelast and model_key_r in tlx2pd_namelast:
                model_key = model_key_l + '.' + tlx2pd_namelast[model_key_r]
            else:
                model_key = model_key
        weights.append(param[model_key])
    assign_weights(weights, model)
    del weights
