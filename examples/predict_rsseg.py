# coding: utf-8
import os
os.environ['TL_BACKEND'] = 'paddle'
import numpy as np
from examples.utils.load_image import load_image

img_path = os.path.join(os.path.dirname(__file__) + "/images/LC80010812013365LGN00_18_photo.png")


def predict_pd(model, model_name="unet"):
    print("=" * 16, "Predict value in forward propagation - PaddlePaddle", "=" * 16)
    print('Model name:', model_name)
    model.eval()

    img = load_image(img_path, mode='pd', size=(256, 256))
    results = model(img)[0]
    print('Predicted value:', results)
    return results


def predict_tlx(model, model_name="unet"):
    print("=" * 16, "Predict value in forward propagation - TensorLayerX", "=" * 16)
    print('Model name:', model_name)
    model.set_eval()

    img = load_image(img_path, mode='tlx', size=(256, 256))
    results = model(img)[0]
    print('Predicted value:', results)
    return results


def calc_diff(result_tlx, result_pd, model_name="unet"):
    print("=" * 16, " Prediction Error ", "=" * 16)
    print('Model name:', model_name)
    diff = np.abs((np.array(result_tlx) - np.array(result_pd)))
    print('diff sum value:', np.sum(diff))
    return diff
