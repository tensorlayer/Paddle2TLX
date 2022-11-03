# coding: utf-8
import os
os.environ['TL_BACKEND'] = 'paddle'
import numpy as np
from utils.load_image import load_image


def predict_pd(model, image_file):
    """ paddlepaddle single sample prediction """
    import paddle

    print('Model name:', f'{model.__class__.__name__}')
    # img = paddle.rand([1, 3, 224, 224])
    img = load_image(image_file, mode="pd")
    print('Input shape:', img.shape)
    out = model(img)

    # probs = np.array(out[0])
    if isinstance(out, paddle.Tensor):
        probs = paddle.nn.Softmax()(out[0]).numpy()
    elif isinstance(out, list):
        probs = paddle.nn.Softmax()(out[0])[0].numpy()
    preds = (np.argsort(probs)[::-1])[0:5]
    file_path = 'images/imagenet_classes.txt'
    with open(file_path) as f:
        class_names = [line.strip() for line in f.readlines()]
    print('Predicted category:', class_names[np.argmax(out[0])])
    for p in preds:
        print(class_names[p], probs[p])


def predict_tlx(model, image_file):
    """ tensorlayerx single sample prediction """
    import paddle
    import tensorlayerx as tlx

    print('Model name:', f'{model.__class__.__name__}')
    # print trainable weights
    # for w in model.trainable_weights:
    #     print(w.name, w.shape)

    img = load_image(image_file, "tlx")
    out = model(img)
    if isinstance(out, paddle.Tensor):  # TODO - replace it
        probs = tlx.ops.softmax(out[0]).numpy()
    elif isinstance(out, list):
        probs = tlx.ops.softmax(out[0])[0].numpy()
    preds = (np.argsort(probs)[::-1])[0:5]

    file_path = 'images/imagenet_classes.txt'
    with open(file_path) as f:
        class_names = [line.strip() for line in f.readlines()]
    print('Predicted category:', class_names[np.argmax(out[0])])
    for p in preds:
        print(class_names[p], probs[p])


def predict_batch(model, image_file):
    pass


def calc_diff(model_tlx, model_pd, image_file, mode_name="VGG16"):
    """ compare prediction error of inference model """
    model_tlx.set_eval()
    img = load_image(image_file, "tlx")
    result_tlx = model_tlx(img)
    # print(result_tlx)
    model_pd.eval()
    img = load_image(image_file, "pd")
    result_pd = model_pd(img)
    # print(result_pd)

    file_path = 'images/imagenet_classes.txt'
    with open(file_path) as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"-------------{mode_name}----------------------")
    print(f'Model {model_tlx.__class__.__name__} predict category - TLX:', classes[np.argmax(result_tlx[0])])
    print(f'Model {model_tlx.__class__.__name__} predict category - Paddle:', classes[np.argmax(result_pd[0])])

    diff = np.fabs(np.array(result_tlx) - np.array(result_pd))/np.abs(np.array(result_tlx))
    print('diff sum value:', np.sum(diff)*100/1000)
    print('diff max value:', np.max(diff))
