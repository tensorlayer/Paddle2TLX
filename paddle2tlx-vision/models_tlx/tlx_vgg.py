# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'tensorflow'
import tensorlayerx.nn as nn
import numpy as np
import paddle
import copy
import tensorlayerx as tlx
from utils.download import get_weights_path_from_url
from utils.load_model import restore_model

__all__ = []

model_urls = {
    'vgg16': ('https://paddle-hapi.bj.bcebos.com/models/vgg16.pdparams',
                 '89bbffc0f87d260be9b8cdc169c991c4'),
    'vgg19': ('https://paddle-hapi.bj.bcebos.com/models/vgg19.pdparams',
                 '23b18bb13d8894f60f54e642be79a0dd')
}


class VGG(nn.Module):
    """VGG model from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        features (nn.Layer): Vgg features create by function make_layers.
        num_classes (int): Output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool): Use pool before the last three fc layer or not. Default: True.
    Examples:
        .. code-block:: python
            from paddle.vision.models import VGG
            from paddle.vision.models.vgg import make_layers
            vgg11_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
            features = make_layers(vgg11_cfg)
            vgg11 = VGG(features)
    """

    def __init__(self, features, num_classes=1000, with_pool=True):
        super(VGG, self).__init__()
        self.features = features
        self.num_classes = num_classes
        self.with_pool = with_pool

        if self.with_pool:
            # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # 20220818
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7), data_format='channels_first')

        if num_classes > 0:
            self.classifier = nn.Sequential(
                # nn.Linear(512 * 7 * 7, 4096),
                nn.Linear(out_features=4096, act=None, in_features=512 * 7 * 7),
                nn.ReLU(),
                nn.Dropout(),
                # nn.Linear(4096, 4096),
                nn.Linear(out_features=4096, act=None, in_features=4096),
                nn.ReLU(),
                nn.Dropout(),
                # nn.Linear(4096, num_classes),
                nn.Linear(out_features=num_classes, act=None, in_features=4096),  # W_init=None, b_init=False
            )

    def forward(self, x):
        # print(self.features[0](x).shape)
        x = self.features(x)
        # print("Conv shape", x.shape)
        if self.with_pool:
            x = self.avgpool(x)
        if self.num_classes > 0:
            # x = paddle.flatten(x, 1)  # predict value is all equal to paddle prediction here
            x = nn.Flatten()(x)
            # print('x.numpy =', x.shape)
            x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0, data_format='channels_first')]  # padding: 0|VALID, 1(默认)|SAME
        else:
            conv2d = nn.Conv2d(out_channels=v, kernel_size=(3, 3), stride=(1, 1), act=None, padding=1,
                               in_channels=in_channels, data_format='channels_first')  # padding: 0|VALID, 1(默认)|SAME - 数字对上就OK
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(num_features=v, data_format='channels_first'), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B':
        [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512,
        512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
        'M', 512, 512, 512, 512, 'M'
    ],
}

pd2tlx = {'features.0.weight': 'features.0.W',
          'features.2.weight': 'features.2.W',
          'features.5.weight': 'features.5.W',
          'features.7.weight': 'features.7.W',
          'features.10.weight': 'features.10.W',
          'features.12.weight': 'features.12.W',
          'features.14.weight': 'features.14.W',
          'features.17.weight': 'features.17.W',
          'features.19.weight': 'features.19.W',
          'features.21.weight': 'features.21.W',
          'features.24.weight': 'features.24.W',
          'features.26.weight': 'features.26.W',
          'features.28.weight': 'features.28.W',
          'features.0.bias': 'features.0.b',
          'features.2.bias': 'features.2.b',
          'features.5.bias': 'features.5.b',
          'features.7.bias': 'features.7.b',
          'features.10.bias': 'features.10.b',
          'features.12.bias': 'features.12.b',
          'features.14.bias': 'features.14.b',
          'features.17.bias': 'features.17.b',
          'features.19.bias': 'features.19.b',
          'features.21.bias': 'features.21.b',
          'features.24.bias': 'features.24.b',
          'features.26.bias': 'features.26.b',
          'features.28.bias': 'features.28.b',
          'classifier.0.weight': 'classifier.0.W',
          'classifier.3.weight': 'classifier.2.W',
          'classifier.6.weight': 'classifier.4.W',
          'classifier.0.bias': 'classifier.0.b',
          'classifier.3.bias': 'classifier.2.b',
          'classifier.6.bias': 'classifier.4.b'}


def get_new_weight(param):
    """ 新增函数，调整参数key """
    new_param = {}
    for key in param.keys():
        new_param[pd2tlx[key]] = param[key]
        print(key, ":", param[key].shape, "vs", pd2tlx[key], ":", new_param[pd2tlx[key]].shape)
    return new_param


def _vgg(arch, cfg, batch_norm, pretrained, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])
        param = paddle.load(weight_path)
        # model.load_dict(param)
        # new_param = get_new_weight(param)
        # model.load_dict(new_param)
        restore_model(param, model)
    return model


def vgg11(pretrained=False, batch_norm=False, **kwargs):
    """VGG 11-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.
    Examples:
        .. code-block:: python
            from paddle.vision.models import vgg11
            # build model
            model = vgg11()
            # build vgg11 model with batch_norm
            model = vgg11(batch_norm=True)
    """
    model_name = 'vgg11'
    if batch_norm:
        model_name += ('_bn')
    return _vgg(model_name, 'A', batch_norm, pretrained, **kwargs)


def vgg13(pretrained=False, batch_norm=False, **kwargs):
    """VGG 13-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.
    Examples:
        .. code-block:: python
            from paddle.vision.models import vgg13
            # build model
            model = vgg13()
            # build vgg13 model with batch_norm
            model = vgg13(batch_norm=True)
    """
    model_name = 'vgg13'
    if batch_norm:
        model_name += ('_bn')
    return _vgg(model_name, 'B', batch_norm, pretrained, **kwargs)


def vgg16(pretrained=False, batch_norm=False, **kwargs):
    """VGG 16-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.
    Examples:
        .. code-block:: python
            from paddle.vision.models import vgg16
            # build model
            model = vgg16()
            # build vgg16 model with batch_norm
            model = vgg16(batch_norm=True)
    """
    model_name = 'vgg16'
    if batch_norm:
        model_name += ('_bn')
    return _vgg(model_name, 'D', batch_norm, pretrained, **kwargs)


def vgg19(pretrained=False, batch_norm=False, **kwargs):
    """VGG 19-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.
    Examples:
        .. code-block:: python
            from paddle.vision.models import vgg19
            # build model
            model = vgg19()
            # build vgg19 model with batch_norm
            model = vgg19(batch_norm=True)
    """
    model_name = 'vgg19'
    if batch_norm:
        model_name += ('_bn')
    return _vgg(model_name, 'E', batch_norm, pretrained, **kwargs)
