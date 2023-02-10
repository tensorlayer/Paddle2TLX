import os
import six
import pickle

WEIGHTS_HOME = os.path.expanduser("~/.cache/paddle/weights")


def get_path_from_url(urls, model_name, model_type=""):
    from .download import get_path_from_url

    model_url = ''
    if isinstance(urls, dict):
        if isinstance(urls[model_name], list) or isinstance(urls[model_name], tuple):
            if len(urls[model_name]) == 2:
                model_url = urls[model_name][0]
            elif len(urls[model_name]) == 1:
                model_url = urls[model_name]
        elif isinstance(urls[model_name], str):
            model_url = urls[model_name]
    elif isinstance(urls, str):
        model_url = urls
    file_name = os.path.split(model_url)[-1]
    file_path = WEIGHTS_HOME
    if model_type:
        if model_type in ["paddleclas", "paddlerscd", "paddlegan", "paddlenlp"]:
            file_path = os.path.join(WEIGHTS_HOME, model_type)  # todo
        elif model_type in ["paddlersseg", "paddleseg"]:
            file_path = os.path.join(WEIGHTS_HOME, model_type, model_name)
    if not os.path.exists(model_url):
        weights_path = get_path_from_url(model_url, file_path)
    else:
        weights_path = model_url

    return model_url, weights_path


def tlx_load(file_name):
    with open(file_name, 'rb') as f:
        state_dicts = pickle.load(f) if six.PY2 else pickle.load(
            f, encoding='latin1')
    return state_dicts


def get_new_key(param):
    # keys_namelast = {
    #     '_batch_norm.weight': 'batch_norm.weight',
    #     '_batch_norm.bias': 'batch_norm.bias',
    #     '_batch_norm._mean': 'batch_norm._mean',
    #     '_batch_norm._variance': 'batch_norm._variance',
    # }  # default key: new key
    new_param = dict()
    for name in param.keys():
        name_l = ".".join(name.split('.')[:-2])
        if '_batch_norm.weight' in name:
            new_key = name_l + "." + "batch_norm.weight"
            new_param[new_key] = param[name]
        elif '_batch_norm.bias' in name:
            new_key = name_l + "." + "batch_norm.bias"
            new_param[new_key] = param[name]
        elif '_batch_norm._mean' in name:
            new_key = name_l + "." + "batch_norm._mean"
            new_param[new_key] = param[name]
        elif '_batch_norm._variance' in name:
            new_key = name_l + "." + "batch_norm._variance"
            new_param[new_key] = param[name]
        else:
            new_param[name] = param[name]
    return new_param
