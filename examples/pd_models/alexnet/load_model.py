# coding: utf-8
import paddle
from paddle.utils.download import get_weights_path_from_url


def load_model(model, arch, model_urls):
    assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
    weight_path = get_weights_path_from_url(model_urls[arch][0],
                                            model_urls[arch][1])

    param = paddle.load(weight_path)
    model.load_dict(param)
    return model
