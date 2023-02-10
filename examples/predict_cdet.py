# coding: utf-8
import os
os.environ['TL_BACKEND'] = 'paddle'
import numpy as np
from examples.utils.load_image import load_image

img_couple = [
    os.path.join(os.path.dirname(__file__), 'images/im1_2.bmp'),
    os.path.join(os.path.dirname(__file__), 'images/im1_1.bmp'),
]

def predict_pd(model, model_name="bit"):
    print("=" * 16, "Predict value in forward propagation - PaddlePaddle", "=" * 16)
    print('Model name:', model_name)
    model.eval()

    img1 = load_image(img_couple[0], mode='pd', size=(256, 256))
    img2 = load_image(img_couple[1], mode='pd', size=(256, 256))
    results = model(img1, img2)[0]
    print('Predicted value:', results)
    return results


def predict_tlx(model, model_name="bit"):
    print("=" * 16, "Predict value in forward propagation - TensorLayerX", "=" * 16)
    print('Model name:', model_name)
    model.set_eval()

    img1 = load_image(img_couple[0], mode='tlx', size=(256, 256))
    img2 = load_image(img_couple[1], mode='tlx', size=(256, 256))
    results = model(img1, img2)[0]
    print('Predicted value:', results)
    return results


def calc_diff(result_tlx, result_pd, model_name="bit"):
    print("=" * 16, " Prediction Error ", "=" * 16)
    print('Model name:', model_name)
    diff = np.abs((np.array(result_tlx) - np.array(result_pd)))
    print('diff sum value:', np.sum(diff))
    return diff
