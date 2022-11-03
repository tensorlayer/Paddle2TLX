# coding: utf-8
import tensorlayerx as tlx
from tensorlayerx.files import assign_weights


# # assign sequential model weights
# def restore_model(param, model):
#     weights = []
#     for val in param.items():
#         weights.append(val[1])
#         if len(model.all_weights) == len(weights):
#             break
#     # assign weight values
#     assign_weights(weights, model)
#     del weights


def restore_model_v1(param, model):
    tlx2pd_namelast = {'filters': 'weight',        # conv2d
                       'biases': 'bias',           # linear
                       'weights': 'weight',        # linear
                       'gamma': 'weight',          # bn
                       'beta': 'bias',             # bn
                       'moving_mean': '_mean',     # bn
                       'moving_var': '_variance',  # bn
                       }
    # print([{i: k} for i, k in model.named_parameters()])
    model_state = [i for i, k in model.named_parameters()]
    weights = []

    for i in range(len(model_state)):
        model_key = model_state[i]
        model_key_s, model_key_e = model_key.rsplit('.', 1)
        if model_key_e in tlx2pd_namelast:
            new_model_state = model_key_s + '.' + tlx2pd_namelast[model_key_e]
            weights.append(param[new_model_state])
        else:
            print(model_key_e)
    assign_weights(weights, model)
    del weights


def restore_model(param, model):
    pd2tlx_namelast = {'filters': 'weight',
                       'gamma': 'weight',
                       'weights': 'weight',
                       'beta': 'bias',
                       'biases': 'bias',
                       'moving_mean': '_mean',
                       'moving_var': '_variance', }
    # print([{i: k} for i, k in model.named_parameters()])
    model_state = [i for i, k in model.named_parameters()]

    # [print(i,k.shape) for i, k in model.named_parameters()]
    weights = []
    for i in range(len(model_state)):
        model_key = model_state[i]
        model_keys = model_key.rsplit('.', 1)
        if len(model_keys) == 2:
            if model_keys[1] in pd2tlx_namelast:
                model_key = model_keys[0] + '.' + pd2tlx_namelast[model_keys[1]]
            else:
                model_key = model_key
                print(model_keys[1])

        weights.append(param[model_key])

    # print(len(model_state), len(param), len(weights))
    assign_weights(weights, model)
    del weights

def get_new_weights(param, model):
    pd2tlx_namelast = {'filters': 'weight',
                       'gamma': 'weight',
                       'weights': 'weight',
                       'beta': 'bias',
                       'biases': 'bias',
                       'moving_mean': '_mean',
                       'moving_var': '_variance', }
    # print([{i: k} for i, k in model.named_parameters()])
    model_state = [i for i, k in model.named_parameters()]

    # [print(i,k.shape) for i, k in model.named_parameters()]
    weights = []
    for i in range(len(model_state)):
        model_key = model_state[i]
        model_keys = model_key.rsplit('.', 1)
        if len(model_keys) == 2:
            if model_keys[1] in pd2tlx_namelast:
                model_key = model_keys[0] + '.' + pd2tlx_namelast[model_keys[1]]
            else:
                model_key = model_key
                print(model_keys[1])

        weights.append(param[model_key])

    # print(len(model_state), len(param), len(weights))
    assign_weights(weights, model)
    del weights