# coding: utf-8
import os
os.environ['TL_BACKEND'] = 'paddle'
import warnings
import six
import wget
import pickle
import paddle
import numpy as np
from tensorlayerx.files import assign_weights
from .load_model import get_path_from_url, tlx_load, get_new_key
warnings.filterwarnings("ignore")


# mode_urls = {
#     'rnn': './params/rnn_best_model_final.pdparams',
#     'lstm': './params/lstm_best_model_final.pdparams',
#     'textcnn': './params/textcnn_best_model_final.pdparams'
# }


def load_model_nlp(model, model_name, urls):
    _, weights_path = get_path_from_url(urls, model_name, "paddlenlp")
    param = paddle.load(weights_path)
    param = get_new_key(param)  # todo
    model.load_dict(param)
    return model


def restore_model_nlp(model, model_name, urls):
    _, weights_path = get_path_from_url(urls, model_name, "paddlenlp")
    param = tlx_load(weights_path)
    restore_model(param, model)
    return model


def restore_model(param, model):
    tlx2pd_namelast = {'filters': 'weight',  # conv2d
                       'biases': 'bias',  # linear
                       'weights': 'weight',  # linear
                       'gamma': 'weight',  # bn
                       'beta': 'bias',  # bn
                       'moving_mean': '_mean',  # bn
                       'moving_var': '_variance',  # bn
                       'embeddings': 'weight',
                       'weight_ih': 'weight_ih_l0',
                       'weight_hh': 'weight_hh_l0',
                       'bias_ih': 'bias_ih_l0',
                       'bias_hh': 'bias_hh_l0',
                       'q_weight': 'q_proj.weight',
                       'k_weight': 'k_proj.weight',
                       'v_weight': 'v_proj.weight',
                       'out_weight': 'out_proj.weight',
                       'q_bias': 'q_proj.bias',
                       'k_bias': 'k_proj.bias',
                       'v_bias': 'v_proj.bias',
                       'out_bias': 'out_proj.bias',
                       }

    model_state = [i for i, k in model.named_parameters()]
    weights = []
    for i in range(len(model_state)):
        model_key = model_state[i]
        model_keys = model_key.rsplit('.', 1)
        if len(model_keys) == 2:
            if model_keys[1] in tlx2pd_namelast:
                model_key = model_keys[0] + '.' + tlx2pd_namelast[model_keys[1]]
            elif 'rnn' in model_keys[0] or 'lstm' in model_keys[0]:
                new_keys = model_keys[0].split('.')
                model_key = new_keys[0] + '.' + tlx2pd_namelast[new_keys[1]]
            else:
                model_key = model_key
        weights.append(param[model_key])

    assign_weights(weights, model)
    del weights
