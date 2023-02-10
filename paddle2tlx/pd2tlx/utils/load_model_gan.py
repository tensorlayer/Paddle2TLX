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


def load_model_gan(model, urls, model_name):
    _, weights_path = get_path_from_url(urls, model_name, "paddlegan")
    param = paddle.load(weights_path)
    restore_model_pd(param, model)
    return model


def restore_model_gan(model, urls, model_name):
    _, weights_path = get_path_from_url(urls, model_name, "paddlegan")
    param = tlx_load(weights_path)
    if model_name in ["cyclegan", "prenet", "stylegan", "stargan"]:
        restore_model_tlx(param, model, model_name=model_name)
    elif model_name == "ugatit":
        param = get_new_weights(param)
        restore_model_tlx(param, model, model_name=model_name)
    return model


def restore_model_pd(state_dicts, model):
    if is_dict_in_dict_weight(state_dicts):
        for net_name, net in model.nets.items():
            if net_name in state_dicts:
                net.set_state_dict(state_dicts[net_name])
                print('Loaded pretrained weight for net {}'.format(net_name))
            else:
                print(
                    'Can not find state dict of net {}. Skip load pretrained weight for net {}'
                        .format(net_name, net_name))
    else:
        assert len(model.nets
                   ) == 1, 'checkpoint only contain weight of one net, but model contains more than one net!'
        net_name, net = list(model.nets.items())[0]
        net.set_state_dict(state_dicts)
        print('Loaded pretrained weight for net {}'.format(net_name))
    return model


def restore_model_tlx(state_dicts, model, model_name):
    if is_dict_in_dict_weight(state_dicts):
        for net_name, net in model.nets.items():
            if net_name in state_dicts:
                if model_name == "ugatit":
                    net.set_state_dict(state_dicts[net_name])
                # assign_weights(state_dicts[net_name], net)
                # net = model.nets['disGB']
                # net = model.nets['disLA']
                elif model_name in ["cyclegan", "prenet", "stylegan", "stargan"]:
                    restore_model(state_dicts[net_name], net)
                print('Loaded pretrained weight for net {}'.format(net_name))
            else:
                print(
                    'Can not find state dict of net {}. Skip load pretrained weight for net {}'
                        .format(net_name, net_name))
    else:
        assert len(model.nets
                   ) == 1, 'checkpoint only contain weight of one net, but model contains more than one net!'
        net_name, net = list(model.nets.items())[0]
        if model_name == "ugatit":
            net.set_state_dict(state_dicts)
        elif model_name in ["cyclegan", "prenet", "stylegan", "stargan"]:
            restore_model(state_dicts, net)
        print('Loaded pretrained weight for net {}'.format(net_name))
    return model


def is_dict_in_dict_weight(state_dict):
    if isinstance(state_dict, dict) and len(state_dict) > 0:
        val = list(state_dict.values())[0]
        if isinstance(val, dict):
            return True
        else:
            return False
    else:
        return False


def get_new_weights(state_dicts):
    specal_name = {'gap_fc.weight': 'gap_fc.weights', # conv2d
                    "FC.0.weight":"FC.0.weights",
                    "FC.2.weight":"FC.2.weights",
                       'gmp_fc.weight': 'gmp_fc.weights',
                       "gamma.weight":"gamma.weights",
                       "beta.weight":"beta.weights"}
    new_state_dicts={}
    for sub_net in state_dicts.keys():
        # tlx train load
        new_state_dicts[sub_net] = {}
        for key in state_dicts[sub_net]:
            if key in specal_name.keys():
                new_state_dicts[sub_net][specal_name[key]] = state_dicts[sub_net][key]
                continue
            if key.endswith('.bias'):
                new_state_dicts[sub_net][key.replace(".bias",".biases")] = state_dicts[sub_net][key]
            elif key.endswith('.scale'):
                new_state_dicts[sub_net][key.replace(".scale",".filters")] = state_dicts[sub_net][key]
            elif key.endswith('.weight'):
                new_state_dicts[sub_net][key.replace(".weight",".filters")] = state_dicts[sub_net][key]
            else:
                new_state_dicts[sub_net][key] = state_dicts[sub_net][key]
    return new_state_dicts


def restore_model(params, model):
    import warnings

    tlx2pd_namelast = {'filters': 'weight',
                       'biases': 'bias',
                       'weights': 'weight',
                       # 'key1_weights': 'weight_orig',
                       # 'key2_filters': 'weight_orig',
                       }
    model_states = [i for i, k in model.named_parameters()]

    weights = []
    for model_k, model_v in model.named_parameters():
        model_key_split = model_k.rsplit('.', 1)
        if len(model_key_split) == 2 and model_key_split[1] in tlx2pd_namelast:
            param_k = model_key_split[0] + '.' + tlx2pd_namelast[model_key_split[1]]
            # assert model_v.shape == params[param_k].shape
            # weights.append(params[param_k])
            # print('----------------------')
            # print(model_k, model_v.shape)
            # print(param_k, params[param_k].shape)
            try:
                weights.append(params[param_k])
                print('----------------------')
                print(model_k, model_v.shape)
                print(param_k, params[param_k].shape)
            except KeyError as err:
                warnings.warn(("Skip loading for {}. ".format(param_k) + str(err)))
                if model_k in params:
                    weights.append(params[model_k])
                elif param_k.endswith("weight"):  # spacial case
                    param_k1 = model_key_split[0] + '.' + 'scale'
                    param_k2 = model_key_split[0] + '.' + 'weight_orig'
                    if param_k1 in params:
                        weights.append(params[param_k1])
                    elif param_k2 in params:
                        weights.append(params[param_k2])
        elif model_k in params:
            param_k = model_k
            # assert model_v.shape == params[param_k].shape
            weights.append(params[param_k])
            print('model_key == param key, key is: ', param_k)
            print(model_k, model_v.shape)
            print(param_k, params[param_k].shape)
        else:
            print('unmatched model key: ', model_k)

    print('len model states:', len(model_states), ' len params:', len(params), 'len matched weights:', len(weights))
    assign_weights(weights, model)
    del weights
    return model
