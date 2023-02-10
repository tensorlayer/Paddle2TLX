# coding: utf-8
import os
import six
import pickle
import paddle
import tensorlayerx as tlx
from tensorlayerx.files import assign_weights
from .load_model import get_path_from_url, tlx_load, get_new_key


def load_model_rsseg(model, model_name, urls):
    _, weights_path = get_path_from_url(urls, model_name, "paddlersseg")
    param = paddle.load(weights_path)
    param = get_new_key(param)  # todo
    model.load_dict(param)
    return model


def restore_model_rsseg(model, model_name, urls):
    _, weights_path = get_path_from_url(urls, model_name, "paddlersseg")
    param = tlx_load(weights_path)
    restore_model(param, model)
    return model


def restore_model(param, model):
    tlx2pd_namelast = {'filters': 'weight',        # conv2d
                       'biases': 'bias',           # linear
                       'weights': 'weight',        # linear
                       'gamma': 'weight',          # bn, ln
                       'beta': 'bias',             # bn, ln
                       'moving_mean': '_mean',     # bn
                       'moving_var': '_variance',  # bn
                       # custom op name
                       'batch_norm.gamma': '_batch_norm.weight',
                       'batch_norm.beta': '_batch_norm.bias',
                       'batch_norm.moving_mean': '_batch_norm._mean',
                       'batch_norm.moving_var': '_batch_norm._variance',
                       }
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
            model_key_r = model_key[indices[-2]+1:]
        if len(model_keys) == 2 and model_key not in param:  # if len(model_keys) == 2:
            if model_keys[1] in tlx2pd_namelast and model_key_r not in tlx2pd_namelast:
                model_key = model_keys[0] + '.' + tlx2pd_namelast[model_keys[1]]
            elif model_keys[1] in tlx2pd_namelast and model_key_r in tlx2pd_namelast:
                model_key = model_key_l + '.' + tlx2pd_namelast[model_key_r]
            else:
                model_key = model_key
        weights.append(param[model_key])
    assign_weights(weights, model)
    # save_file = os.path.join(PRETRAINED_PATH_TLX, arch + ".npz")
    # tlx.files.save_npz(model.all_weights, name=save_file)  # save model
    del weights
