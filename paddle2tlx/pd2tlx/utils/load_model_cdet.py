# coding: utf-8
import os
os.environ['TL_BACKEND'] = 'paddle'
import six
import pickle
import os
import wget
import paddle
import tensorlayerx as tlx
from tensorlayerx.files import assign_weights
from .load_model import get_path_from_url, tlx_load


# mode_urls = {"bit":"https://paddlers.bj.bcebos.com/pretrained/cd/levircd/weights/bit_levircd.pdparams",
#             "cdnet": "https://paddlers.bj.bcebos.com/pretrained/cd/levircd/weights/cdnet_levircd.pdparams",
#             "dsamnet":"https://paddlers.bj.bcebos.com/pretrained/cd/levircd/weights/dsamnet_levircd.pdparams",
#             "stanet":"https://paddlers.bj.bcebos.com/pretrained/cd/levircd/weights/stanet_levircd.pdparams",
#             "dsifn":"https://paddlers.bj.bcebos.com/pretrained/cd/levircd/weights/dsifn_levircd.pdparams",
#             "snunet":"https://paddlers.bj.bcebos.com/pretrained/cd/levircd/weights/snunet_levircd.pdparams",
#             "fcef":"https://paddlers.bj.bcebos.com/pretrained/cd/levircd/weights/fc_ef_levircd.pdparams",
#             "fccdn":"https://paddlers.bj.bcebos.com/pretrained/cd/levircd/weights/fccdn_levircd.pdparams",
#             "resnet18": "https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams",
#             "resnet34": "https://paddle-hapi.bj.bcebos.com/models/resnet34.pdparams",
#             "resnet50": "https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams",
#             "resnet101": "https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams",
#             "resnet152": "https://paddle-hapi.bj.bcebos.com/models/resnet152.pdparams",
#             "vgg16": "https://paddle-hapi.bj.bcebos.com/models/vgg16.pdparams",
#             "vgg19": "https://paddle-hapi.bj.bcebos.com/models/vgg19.pdparams"}


def get_new_weights(model_dict):
    new_weights = {}
    for key in model_dict:
        if key.endswith('.1.fn.fn.3.weight'):
            new_key = key.replace(".1.fn.fn.3.weight", ".1.fn.fn.2.weight")
            new_weights[new_key] = model_dict[key]
        elif key.endswith('.1.fn.fn.3.bias'):
            new_key = key.replace(".1.fn.fn.3.bias", ".1.fn.fn.2.bias")
            new_weights[new_key] = model_dict[key]
        else:
            new_weights[key] = model_dict[key]
    return new_weights


def load_model_cdet(model, urls, model_name):
    _, weights_path = get_path_from_url(urls, model_name, "paddlerscd")
    param = paddle.load(weights_path)
    new_weights = get_new_weights(param)
    model.load_dict(new_weights)
    return model


def restore_model_cdet(model, urls, model_name):
    _, weights_path = get_path_from_url(urls, model_name, "paddlerscd")
    param = tlx_load(weights_path)
    new_weights = get_new_weights(param)
    restore_model(new_weights, model)
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
    # print("*"*45)
    # print(f"param.keys()={param.keys()}")
    # print("="*45)
    # print(f"model_state={model_state}")
    # [print(i,k.shape) for i, k in model.named_parameters()]
    weights = []
    for i in range(len(model_state)):
        model_key = model_state[i]
        if model_key == "enc_pos_embedding":
            weights.append(param[model_key])
            continue
        if model_key.startswith("TDec_x2.dense_1x."):
            model_key = model_key.replace("TDec_x2.dense_1x.", "TDec_x2.dense_1x.0.")
            # weights.append(param[model_key])
            # continue
        if model_key.startswith("TDec_x2.dense_2x."):
            model_key = model_key.replace("TDec_x2.dense_2x.", "TDec_x2.dense_2x.0.")
            # weights.append(param[model_key])
            # continue
        model_keys = model_key.rsplit('.', 1)
        if len(model_keys) == 2:
            if model_keys[1] in pd2tlx_namelast:
                model_key = model_keys[0] + '.' + pd2tlx_namelast[model_keys[1]]
            else:
                model_key = model_key
                print(model_keys[1])
        # print(model_key, param[model_key].shape)
        weights.append(param[model_key])
    # print(len(model_state), len(param), len(weights))
    assign_weights(weights, model)
    del weights
