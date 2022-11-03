# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

# reference: https://arxiv.org/abs/1611.05431

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TL_BACKEND'] = 'paddle'
import numpy as np
import paddle
import tensorlayerx as tlx
# from paddle import ParamAttr
# import paddle.nn as nn
import tensorlayerx.nn as nn
# import paddle.nn.functional as F
# from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from tensorlayerx.nn import Conv2d,BatchNorm, Linear, GroupConv2d
# from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from tensorlayerx.nn import AdaptiveAvgPool2d,MaxPool2d
# from paddle.nn.initializer import Uniform
# from tensorlayerx.nn.initializers import RandomUniform
from utils.download import get_weights_path_from_url, get_path_from_url

from utils.load_model import restore_model
import math

# from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "ResNeXt50_32x4d":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_32x4d_pretrained.pdparams",
    "ResNeXt50_64x4d":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_64x4d_pretrained.pdparams",
    "ResNeXt101_32x4d":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x4d_pretrained.pdparams",
    "ResNeXt101_64x4d":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_64x4d_pretrained.pdparams",
    "ResNeXt152_32x4d":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_32x4d_pretrained.pdparams",
    "ResNeXt152_64x4d":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_64x4d_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


class ConvBNLayer(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None,
                 data_format="channels_first"):
        super(ConvBNLayer, self).__init__()
        self._conv = GroupConv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            n_group=groups,
            # weight_attr=ParamAttr(name=name + "_weights"),
            b_init=None,
            data_format=data_format)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self._batch_norm = BatchNorm(
            num_features=num_filters,
            act=act,
            # param_attr=ParamAttr(name=bn_name + '_scale'),
            # bias_attr=ParamAttr(bn_name + '_offset'),
            # moving_mean_name=bn_name + '_mean',
            # moving_variance_name=bn_name + '_variance',
            data_format=data_format)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 cardinality,
                 shortcut=True,
                 name=None,
                 data_format="channels_first"):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a",
            data_format=data_format)
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            groups=cardinality,
            stride=stride,
            act='relu',
            name=name + "_branch2b",
            data_format=data_format)
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 2 if cardinality == 32 else num_filters,
            filter_size=1,
            act=None,
            name=name + "_branch2c",
            data_format=data_format)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 2
                if cardinality == 32 else num_filters,
                filter_size=1,
                stride=stride,
                name=name + "_branch1",
                data_format=data_format)

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = tlx.add(value=short, bias=conv2)
        y = tlx.relu(y)
        return y


class ResNeXt(nn.Module):
    def __init__(self,
                 layers=50,
                 class_num=1000,
                 cardinality=32,
                 input_image_channel=3,
                 data_format="channels_first"):
        super(ResNeXt, self).__init__()

        self.layers = layers
        self.data_format = data_format
        self.input_image_channel = input_image_channel
        self.cardinality = cardinality
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)
        supported_cardinality = [32, 64]
        assert cardinality in supported_cardinality, \
            "supported cardinality is {} but input cardinality is {}" \
            .format(supported_cardinality, cardinality)
        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_channels = [64, 256, 512, 1024]
        num_filters = [128, 256, 512,
                       1024] if cardinality == 32 else [256, 512, 1024, 2048]

        self.conv = ConvBNLayer(
            num_channels=self.input_image_channel,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name="res_conv1",
            data_format=self.data_format)
        self.pool2d_max = MaxPool2d(
            kernel_size=3, stride=2, padding=1, data_format=self.data_format)

        self.block_list = []
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                if layers in [101, 152] and block == 2:
                    if i == 0:
                        conv_name = "res" + str(block + 2) + "a"
                    else:
                        conv_name = "res" + str(block + 2) + "b" + str(i)
                else:
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels[block] if i == 0 else
                        num_filters[block] * int(64 // self.cardinality),
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        cardinality=self.cardinality,
                        shortcut=shortcut,
                        name=conv_name,
                        data_format=self.data_format))
                self.block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = AdaptiveAvgPool2d(1, data_format=self.data_format)

        self.pool2d_avg_channels = num_channels[-1] * 2

        stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)

        self.out = Linear(
            in_features=self.pool2d_avg_channels,
            out_features=class_num,
            # weight_attr=ParamAttr(
            #     initializer=Uniform(-stdv, stdv), name="fc_weights"),
            # weight_attr=initializer(Uniform(-stdv, stdv)),
            # bias_attr=ParamAttr(name="fc_offset")
            )

    def forward(self, inputs):
        # with paddle.static.amp.fp16_guard():
        # if self.data_format == "NHWC":
        #     inputs = paddle.tensor.transpose(inputs, [0, 2, 3, 1])
        #     inputs.stop_gradient = True
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for block in self.block_list:
            y = block(y)
        # print(y)
        y = self.pool2d_avg(y)
        y = tlx.reshape(y, shape=[-1, self.pool2d_avg_channels])
        y = self.out(y)
        return y

# def restore_model(param, model):
#     from tensorlayerx.files import assign_weights
#     pd2tlx_namelast = {'filters': 'weight',
#                        'gamma': 'weight',
#                        'weights': 'weight',
#                        'beta': 'bias',
#                        'biases': 'bias',
#                        'moving_mean': '_mean',
#                        'moving_var': '_variance', }
#     # print([{i: k} for i, k in model.named_parameters()])
#     model_state = [i for i, k in model.named_parameters()]
#     weights = []
#
#     for i in range(len(model_state)):
#         model_key = model_state[i]
#         model_key_s, model_key_e = model_key.rsplit('.', 1)
#         if model_key_e in pd2tlx_namelast:
#             new_model_state = model_key_s + '.' + pd2tlx_namelast[model_key_e]
#             weights.append(param[new_model_state])
#         else:
#             print(model_key_e)
#     assign_weights(weights, model)
#     del weights

def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    import pickle
    if pretrained is False:
        pass
    elif pretrained is True:
        weight_path = get_weights_path_from_url(model_url)
        # print(weight_path)
        # param = paddle.load(weight_path)
        with open(weight_path, 'rb') as f:
            param = pickle.load(f, encoding='latin1')
        restore_model(param, model)

        # model.save_weights('../../model/resnext_model.npz')
        # param = tlx.files.load_npz(path='../../model/',name='resnext_model.npz')
        # # print(param)
        # from tensorlayerx.files import assign_weights
        # assign_weights(param, model)
        # del param


def ResNeXt50_32x4d(pretrained=False, use_ssld=False, **kwargs):
    model = ResNeXt(layers=50, cardinality=32, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ResNeXt50_32x4d"], use_ssld=use_ssld)
    return model


def ResNeXt50_64x4d(pretrained=False, use_ssld=False, **kwargs):
    model = ResNeXt(layers=50, cardinality=64, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ResNeXt50_64x4d"], use_ssld=use_ssld)
    return model


def ResNeXt101_32x4d(pretrained=False, use_ssld=False, **kwargs):
    model = ResNeXt(layers=101, cardinality=32, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ResNeXt101_32x4d"], use_ssld=use_ssld)
    return model


def ResNeXt101_64x4d(pretrained=False, use_ssld=False, **kwargs):
    model = ResNeXt(layers=101, cardinality=64, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ResNeXt101_64x4d"], use_ssld=use_ssld)
    return model


def ResNeXt152_32x4d(pretrained=False, use_ssld=False, **kwargs):
    model = ResNeXt(layers=152, cardinality=32, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ResNeXt152_32x4d"], use_ssld=use_ssld)
    return model


def ResNeXt152_64x4d(pretrained=False, use_ssld=False, **kwargs):
    model = ResNeXt(layers=152, cardinality=64, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ResNeXt152_64x4d"], use_ssld=use_ssld)
    return model



# import numpy as np
from PIL import Image
def load_image_nchw(image_path):
    """ data format: nchw """
    img = Image.open(image_path)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)
    # PIL打开图片存储顺序为H(高度)，W(宽度)，C(通道)。PaddlePaddle要求数据顺序为CHW，所以需要转换顺序。
    img = img.transpose((2, 0, 1))  # CHW
    # CIFAR训练图片通道顺序为B(蓝),G(绿),R(红), 而PIL打开图片默认通道顺序为RGB,因为需要交换通道。
    img = img[(2, 1, 0), :, :]  # BGR
    img = np.expand_dims(img, 0)
    # # img = img.flatten()
    img = img / 255.0
    # img = to_tensor(img)
    img = tlx.convert_to_tensor(img)
    return img
if __name__ == "__main__":


    model = ResNeXt101_64x4d(pretrained=True)
    # model = resnet18(pretrained=False)
    model.set_eval()
    # for w in model.trainable_weights:
        # print(w.name, w.shape)
    #     print(w)

    x = load_image_nchw("../../images/dog.jpeg")
    print(x.shape)
    # print(x)
    out = model(x)

    file_path = '../../images/imagenet_classes.txt'
    with open(file_path) as f:
        classes = [line.strip() for line in f.readlines()]
    print(classes[np.argmax(out[0])])