# coding: utf-8
import unittest
import numpy as np


class PaddleTLXOpsCmpTest(unittest.TestCase):
    def test_conv2d(self):
        import paddle
        import paddle.nn as nn
        sample_data = np.random.random(size=(1, 3, 224, 224)).astype('float32')
        img = paddle.to_tensor(sample_data)
        conv2d = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)
        paddle_weights = conv2d.weight  # get paddle weights
        out_pd = conv2d(img)
        print(out_pd.numpy().shape)
        print(out_pd.numpy())

        import os
        os.environ['TL_BACKEND'] = 'paddle'
        import tensorlayerx as tlx
        img = tlx.convert_to_tensor(sample_data)
        # img = tlx.nn.Input([1, 400, 400, 3], name='input')
        conv2d = tlx.nn.Conv2d(out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1, b_init=None,
                               in_channels=3, data_format='channels_first')
        conv2d.filters = paddle_weights
        out_tlx = conv2d(img)
        print(out_tlx.numpy().shape)
        print(out_tlx.numpy)

        print("comparison", out_pd.numpy() - out_tlx.numpy())

    def test_linear(self):
        import paddle
        import paddle.nn as nn
        sample_data = np.random.random(size=100).astype('float32')
        data = paddle.to_tensor(sample_data)
        layer = nn.Linear(100, 10)  # imitate prediction of classification model
        paddle_weights = layer.weight
        out_pd = layer(data)
        print(out_pd.numpy())

        import os
        os.environ['TL_BACKEND'] = 'paddle'
        import tensorlayerx as tlx
        data = tlx.convert_to_tensor(sample_data)
        layer = tlx.nn.Linear(out_features=10, in_features=100)
        layer.weights = paddle_weights
        out_tlx = layer(data)
        print(out_tlx.numpy())

        print("comparison", out_pd.numpy() - out_tlx.numpy())

    def test_BatchNorm2d(self):
        import paddle
        np.random.seed(123)
        sample_data = np.random.random(size=(1, 3, 64, 64)).astype('float32')
        x = paddle.to_tensor(sample_data)
        batch_norm = paddle.nn.BatchNorm2D(num_features=3)
        paddle_weights = batch_norm.weight
        out_pd = batch_norm(x)
        print(out_pd)

        import os
        os.environ['TL_BACKEND'] = 'paddle'
        import tensorlayerx as tlx
        # in static model, no need to specify num_features
        # x = tlx.nn.Input([1, 3, 64, 64], name='input')
        x = tlx.convert_to_tensor(sample_data)
        layer = tlx.nn.BatchNorm2d(data_format='channels_first')
        layer.weights = paddle_weights
        out_tlx_static = layer(x)
        print(out_tlx_static)
        # in dynamic model, build by specifying num_features
        layer = tlx.nn.BatchNorm2d(num_features=3, data_format='channels_first')
        layer.weights = paddle_weights
        out_tlx_dynamic = layer(x)
        print(out_tlx_dynamic)


if __name__ == '__main__':
    unittest.main()
