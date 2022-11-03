# coding: utf-8
import os
os.environ['TL_BACKEND'] = 'paddle'  # config paddle as backend in first
import numpy as np
from examples.utils.load_image import load_image


def predict_pd(model, image_file, model_name="vgg16"):
    """ paddlepaddle sample prediction """
    import paddle

    print("=" * 16, " PaddlePaddle ", "=" * 16)
    model.eval()
    # print('Model name:', f'{model.__class__.__name__}')
    print('Model name:', model_name)
    # img = paddle.rand([1, 3, 224, 224])
    img = load_image(image_file, mode="pd")
    # print('Input shape:', img.shape)
    out = model(img)

    # probs = np.array(out[0])
    if isinstance(out, paddle.Tensor):
        probs = paddle.nn.Softmax()(out[0]).numpy()
    elif isinstance(out, list):
        probs = paddle.nn.Softmax()(out[0])[0].numpy()
    preds = (np.argsort(probs)[::-1])[0:5]

    file_path = os.path.join(os.path.dirname(__file__), 'images/imagenet_classes.txt')
    with open(file_path) as f:
        class_names = [line.strip() for line in f.readlines()]
    print('Predicted category:', class_names[np.argmax(out[0])])
    for p in preds:
        print(class_names[p], probs[p])


def predict_tlx(model, image_file, model_name="vgg16"):
    """ tensorlayerx sample prediction """
    import paddle
    import tensorlayerx as tlx

    print("="*16, " TensorLayerX ", "="*16)
    model.set_eval()
    print('Model name:', model_name)
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

    file_path = os.path.join(os.path.dirname(__file__), 'images/imagenet_classes.txt')
    with open(file_path) as f:
        class_names = [line.strip() for line in f.readlines()]
    print('Predicted category:', class_names[np.argmax(out[0])])
    for p in preds:
        print(class_names[p], probs[p])


def calc_diff(model_tlx, model_pd, image_file, model_name="vgg16"):
    """ compare prediction error of inference model """
    print("="*16, " Prediction Error ", "="*16)
    model_tlx.set_eval()
    img = load_image(image_file, "tlx")
    result_tlx = model_tlx(img)
    # print(result_tlx)
    model_pd.eval()
    img = load_image(image_file, "pd")
    result_pd = model_pd(img)
    # print(result_pd)

    file_path = os.path.join(os.path.dirname(__file__), 'images/imagenet_classes.txt')
    with open(file_path) as f:
        classes = [line.strip() for line in f.readlines()]
    print(f'Model {model_name} predict category - TLX:', classes[np.argmax(result_tlx[0])])
    print(f'Model {model_name} predict category - Paddle:', classes[np.argmax(result_pd[0])])

    diff = np.fabs(np.array(result_tlx) - np.array(result_pd))
    print('diff sum value:', np.sum(diff))
    print('diff max value:', np.max(diff))
