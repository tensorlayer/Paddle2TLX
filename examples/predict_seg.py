# coding: utf-8
import os
os.environ['TL_BACKEND'] = 'paddle'
import numpy as np
from examples.utils.load_image import load_image
from paddle2tlx.pd2tlx.utils import load_model_seg, restore_model_seg

img_path = os.path.join(os.path.dirname(__file__) + "/images/LC80010812013365LGN00_18_photo.png")
models_urls = {
    'fastfcn': 'https://bj.bcebos.com/paddleseg/dygraph/ade20k/fastfcn_resnet50_os8_ade20k_480x480_120k/model.pdparams',
    'fast_scnn': 'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fastscnn_cityscapes_1024x1024_160k/model.pdparams',
    'enet': 'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/enet_cityscapes_1024x512_80k/model.pdparams',
    'hrnet': 'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/HRNet_W48_contrast_cityscapes_1024x512_60k/model.pdparams',
    'encnet': 'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/encnet_resnet101_os8_cityscapes_1024x512_80k/model.pdparams',
    'bisenet': 'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/bisenetv1_resnet18_os8_cityscapes_1024x512_160k/model.pdparams'
}


def predict_pd(model, model_name="fastfcn"):
    print("=" * 16, "Predict value in forward propagation - PaddlePaddle", "=" * 16)
    print('Model name:', model_name)
    model.eval()

    img = load_image(img_path, mode='pd', size=(512, 683))
    load_model_seg(model, model_name, models_urls[model_name])
    results = model(img)[0]
    print('Predicted value:', results)
    return results


def predict_tlx(model, model_name="fastfcn"):
    print("=" * 16, "Predict value in forward propagation - TensorLayerX", "=" * 16)
    print('Model name:', model_name)
    model.set_eval()

    img = load_image(img_path, mode='tlx', size=(512, 683))
    restore_model_seg(model, model_name, models_urls[model_name])
    results = model(img)[0]
    print('Predicted value:', results)
    return results


def calc_diff(result_tlx, result_pd, model_name="fastfcn"):
    print("=" * 16, " Prediction Error ", "=" * 16)
    print('Model name:', model_name)
    diff = np.abs((np.array(result_tlx) - np.array(result_pd)))
    print('diff sum value:', np.sum(diff))
    return diff
