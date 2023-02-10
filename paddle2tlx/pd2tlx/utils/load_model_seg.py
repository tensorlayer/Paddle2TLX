# coding: utf-8
import os
import six
import pickle
import paddle
import tensorlayerx as tlx
from tensorlayerx.files import assign_weights
from .load_model import get_path_from_url, tlx_load, get_new_key


def load_model_seg(model, model_name, urls):
    _, weights_path = get_path_from_url(urls, model_name, "paddleseg")
    param = paddle.load(weights_path)
    param = get_new_key(param)  # todo
    model.load_dict(param)
    return model


def restore_model_seg(model, model_name, urls):
    _, weights_path = get_path_from_url(urls, model_name, "paddleseg")
    param = tlx_load(weights_path)
    param = get_new_key(param)
    restore_model(param, model)
    return model


def restore_model(param, model):
    pd2tlx_namelast = {'filters': 'weight',
                       'gamma': 'weight',
                       'weights': 'weight',
                       'beta': 'bias',
                       'biases': 'bias',
                       'moving_mean': '_mean',
                       'moving_var': '_variance',
                       'alpha': '_weight'}
    model_state = [i for i, k in model.named_parameters()]
    weights = []
    for i in range(len(model_state)):
        model_key = model_state[i]
        if model_key == "enc_pos_embedding":
            weights.append(param[model_key])
            continue
        if model_key.startswith("TDec_x2.dense_1x."):
            model_key = model_key.replace("TDec_x2.dense_1x.", "TDec_x2.dense_1x.0.")
        if model_key.startswith("TDec_x2.dense_2x."):
            model_key = model_key.replace("TDec_x2.dense_2x.", "TDec_x2.dense_2x.0.")
        model_keys = model_key.rsplit('.', 1)
        if len(model_keys) == 2:
            if model_keys[1] in pd2tlx_namelast:
                model_key = model_keys[0] + '.' + pd2tlx_namelast[model_keys[1]]
            else:
                model_key = model_key
                print(model_keys[1])
        weights.append(param[model_key])
    assign_weights(weights, model)
    del weights
