# coding: utf-8
import os
import tensorlayerx as tlx
import numpy as np
from PIL import Image

img_path1 = os.path.join(os.path.dirname(__file__), 'images/a.jpg')
img_path2 = os.path.join(os.path.dirname(__file__), 'images/b.jpg')


def load_image(image_path, size=224):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = img / 255.0
    img = tlx.convert_to_tensor(img)
    return img


def predict_tlx(model_tlx, model_name="cyclegan"):
    import tensorlayerx as tlx

    print('Model name:', model_name)
    real_a = load_image(img_path1, size=256)
    real_b = load_image(img_path2, size=256)
    inputs = {'A': real_a, 'B': real_b, 'C': tlx.convert_to_tensor([0])}

    results = []
    # model_pd.eval()
    if model_name == "cyclegan":
        sel_result = []
        model_tlx.setup_input(inputs)
        result = model_tlx.forward()
        for key in ['fake_B', 'fake_A', 'rec_A', 'rec_B']:
            if key in result:
                sel_result.append(result[key])
        results = sel_result
    elif model_name == "ugatit":
        sel_result = dict({})
        model_tlx.setup_input(inputs)
        result = model_tlx.forward()
        for key in ["fake_A2B", "fake_B2A"]:
            if key in result:
                sel_result[key] = result[key]
        results = sel_result
    elif model_name == "stargan":
        # x_src = inputs['A']
        # x_ref = inputs['B']
        # y_ref = inputs['C']
        model_tlx.setup_input(inputs)
        results = model_tlx.forward(inputs)
        pass
    elif model_name == "prenet":
        results = model_tlx.forward(inputs['A'])
    elif model_name == 'stylegan':
        model = model_tlx.nets['disc']
        # model.set_eval()
        results = model(inputs['A'].detach())
    print('Predicted value:', results)
    return results


def predict_pd(model_pd, model_name="cyclegan"):
    import paddle

    print('Model name:', model_name)
    real_a = load_image(img_path1, size=256)
    real_b = load_image(img_path2, size=256)
    inputs = {'A': real_a, 'B': real_b, 'C': paddle.to_tensor([0])}

    results = []
    # model_pd.eval()
    if model_name == "cyclegan":
        sel_result = []
        model_pd.setup_input(inputs)
        result = model_pd.forward()
        for key in ['fake_B', 'fake_A', 'rec_A', 'rec_B']:
            if key in result:
                sel_result.append(result[key])
        results = sel_result
    elif model_name == "ugatit":
        sel_result = dict({})
        model_pd.setup_input(inputs)
        result = model_pd.forward()
        for key in ["fake_A2B", "fake_B2A"]:
            if key in result:
                sel_result[key] = result[key]
        results = sel_result
    elif model_name == "stargan":
        # x_src = inputs['A']
        # x_ref = inputs['B']
        # y_ref = inputs['C']
        model_pd.setup_input(inputs)
        results = model_pd.forward(inputs)
        pass
    elif model_name == "prenet":
        results = model_pd.forward(inputs['A'])
    elif model_name == 'stylegan':
        model = model_pd.nets['disc']
        # model.set_eval()
        results = model(inputs['A'].detach())
    print('Predicted value:', results)
    return results


def calc_diff(result_tlx, result_pd, model_name="rnn"):
    print("=" * 16, " Prediction Error ", "=" * 16)
    print('Model name:', model_name)
    diff = np.abs((np.array(result_tlx) - np.array(result_pd)))
    print('diff sum value:', np.sum(diff))
    return diff
