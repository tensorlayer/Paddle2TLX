# coding: utf-8
import os
import paddle
import tensorlayerx as tlx
from paddle.utils.download import get_weights_path_from_url
from tensorlayerx.files import assign_weights
from paddle2tlx.common.config import PRETRAINED_PATH_TLX, MODEL_PD_URLS


def restore_model(model, arch, load_direct=False):
    """

    :param model: tlx_model
    :param arch: model name
    :param save_path: save path for converted model
    :return:
    """
    if load_direct:
        tlx.files.load_and_assign_npz(f"{PRETRAINED_PATH_TLX}/{arch}.npz", network=model)
        return model

    if len(MODEL_PD_URLS[arch]) > 1:
        weight_path = get_weights_path_from_url(MODEL_PD_URLS[arch][0], MODEL_PD_URLS[arch][1])
    else:
        weight_path = get_weights_path_from_url(MODEL_PD_URLS[arch][0])
    param = paddle.load(weight_path)

    tlx2pd_namelast = {'filters': 'weight',        # conv2d
                       'biases': 'bias',           # linear
                       'weights': 'weight',        # linear
                       'gamma': 'weight',          # bn, ln
                       'beta': 'bias',             # bn, ln
                       'moving_mean': '_mean',     # bn
                       'moving_var': '_variance',  # bn
                       }
    model_state = [i for i, k in model.named_parameters()]
    weights = []
    # get_param_tlx(model)

    for i in range(len(model_state)):
        model_key = model_state[i]
        if model_key.find('.') != -1:
            model_key_s, model_key_e = model_key.rsplit('.', 1)
            if model_key_e in tlx2pd_namelast:
                new_model_state = model_key_s + '.' + tlx2pd_namelast[model_key_e]
                weights.append(param[new_model_state])
            else:
                print(model_key_e)
        else:
            weights.append(param[model_key])
    assign_weights(weights, model)
    save_file = os.path.join(PRETRAINED_PATH_TLX, arch + ".npz")
    tlx.files.save_npz(model.all_weights, name=save_file)  # save model
    del weights
    return model


# def load_model(model, arch, model_path=PRETRAINED_PATH_TLX):
#     """
#
#     :param model:
#     :param arch:
#     :param model_path:
#     :return:
#     """
#     arch = "VGG"
#     tlx.files.load_and_assign_npz(f"{model_path}/{arch}.npz", network=model)
#     return model
