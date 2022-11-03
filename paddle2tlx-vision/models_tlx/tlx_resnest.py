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

# Code was based on https://github.com/zhanghang1989/ResNeSt
# reference: https://arxiv.org/abs/2004.08955

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TL_BACKEND'] = 'paddle'
import numpy as np
import paddle
import math
import tensorlayerx as tlx
# import paddle.nn as nn
import tensorlayerx.nn as nn
# import paddle.nn.functional as F
# from paddle import ParamAttr
# from paddle.nn.initializer import KaimingNormal
from tensorlayerx.nn.initializers import Initializer
# from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from tensorlayerx.nn import Conv2d, BatchNorm, Linear, GroupConv2d
# from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from tensorlayerx.nn import AdaptiveAvgPool2d,MaxPool2d, AvgPool2d
from paddle.regularizer import L2Decay
from utils.download import get_weights_path_from_url, get_path_from_url

from utils.load_model import restore_model
from collections import OrderedDict

MODEL_URLS = {
    "ResNeSt50_fast_1s1x64d":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeSt50_fast_1s1x64d_pretrained.pdparams",
    "ResNeSt50":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeSt50_pretrained.pdparams",
    "ResNeSt101":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeSt101_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


class ConvBNLayer(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()

        bn_decay = 0.0

        self._conv = GroupConv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            dilation=dilation,
            n_group=groups,
            data_format='channels_first',
            # weight_attr=ParamAttr(name=name + "_weight"),
            b_init=None)
        self._batch_norm = BatchNorm(
            num_features=num_filters,
            act=act,
            data_format='channels_first',
            # param_attr=ParamAttr(
            #     name=name + "_scale", regularizer=L2Decay(bn_decay)),
            # bias_attr=ParamAttr(
            #     name + "_offset", regularizer=L2Decay(bn_decay)),
            # moving_mean_name=name + "_mean",
            # moving_variance_name=name + "_variance"
            )

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class rSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(rSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        cardinality = self.cardinality
        radix = self.radix

        batch, r, h, w = x.shape
        if self.radix > 1:
            x = tlx.reshape(
                x,
                shape=[
                    batch, cardinality, radix,
                    int(r * h * w / cardinality / radix)
                ])
            # x = paddle.transpose(x=x, perm=[0, 2, 1, 3])
            # x = nn.functional.softmax(x, axis=1)
            # x = paddle.reshape(x=x, shape=[batch, r * h * w, 1, 1])
            x = tlx.transpose(x, perm=[0, 2, 1, 3])
            x = tlx.softmax(x, axis=1)
            x = tlx.reshape(x, shape=[batch, r * h * w, 1, 1])
        else:
            x = tlx.ops.sigmoid(x)
        return x


class SplatConv(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 radix=2,
                 reduction_factor=4,
                 rectify_avg=False,
                 name=None):
        super(SplatConv, self).__init__()

        self.radix = radix

        self.conv1 = ConvBNLayer(
            num_channels=in_channels,
            num_filters=channels * radix,
            filter_size=kernel_size,
            stride=stride,
            groups=groups * radix,
            act="relu",
            name=name + "_1_weights")

        self.avg_pool2d = AdaptiveAvgPool2d(1, data_format='channels_first')

        inter_channels = int(max(in_channels * radix // reduction_factor, 32))

        # to calc gap
        self.conv2 = ConvBNLayer(
            num_channels=channels,
            num_filters=inter_channels,
            filter_size=1,
            stride=1,
            groups=groups,
            act="relu",
            name=name + "_2_weights")

        # to calc atten
        self.conv3 = GroupConv2d(
            in_channels=inter_channels,
            out_channels=channels * radix,
            kernel_size=1,
            stride=1,
            padding='VALID',
            n_group=groups,
            data_format='channels_first',
            # weight_attr=ParamAttr(
            #     name=name + "_weights", initializer=KaimingNormal()),
            b_init=False)

        self.rsoftmax = rSoftmax(radix=radix, cardinality=groups)

    def forward(self, x):
        x = self.conv1(x)

        if self.radix > 1:
            splited = paddle.split(x, num_or_sections=self.radix, axis=1)
            # gap = paddle.add_n(splited)
            # splited = tlx.ops.split(x, self.radix, axis=1)
            # splited = tlx.convert_to_tensor(splited)
            gap = tlx.add_n(splited)
        else:
            gap = x

        gap = self.avg_pool2d(gap)
        gap = self.conv2(gap)

        atten = self.conv3(gap)
        atten = self.rsoftmax(atten)

        if self.radix > 1:
            attens = paddle.split(atten, self.radix, axis=1)
            # attens = tlx.split(atten, self.radix, axis=1)
            y = tlx.add_n([
                tlx.multiply(split, att)
                for (att, split) in zip(attens, splited)
            ])
        else:
            y = tlx.multiply(x, atten)

        return y


class BottleneckBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 radix=1,
                 cardinality=1,
                 bottleneck_width=64,
                 avd=False,
                 avd_first=False,
                 dilation=1,
                 is_first=False,
                 rectify_avg=False,
                 last_gamma=False,
                 avg_down=False,
                 name=None):
        super(BottleneckBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.radix = radix
        self.cardinality = cardinality
        self.avd = avd
        self.avd_first = avd_first
        self.dilation = dilation
        self.is_first = is_first
        self.rectify_avg = rectify_avg
        self.last_gamma = last_gamma
        self.avg_down = avg_down

        group_width = int(planes * (bottleneck_width / 64.)) * cardinality

        self.conv1 = ConvBNLayer(
            num_channels=self.inplanes,
            num_filters=group_width,
            filter_size=1,
            stride=1,
            groups=1,
            act="relu",
            name=name + "_conv1")

        if avd and avd_first and (stride > 1 or is_first):
            self.avg_pool2d_1 = AvgPool2d(
                kernel_size=3, stride=stride,
                data_format='channels_first', padding=1)

        if radix >= 1:
            self.conv2 = SplatConv(
                in_channels=group_width,
                channels=group_width,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
                radix=radix,
                rectify_avg=rectify_avg,
                name=name + "_splat")
        else:
            self.conv2 = ConvBNLayer(
                num_channels=group_width,
                num_filters=group_width,
                filter_size=3,
                stride=1,
                dilation=dilation,
                groups=cardinality,
                act="relu",
                name=name + "_conv2")

        if avd and avd_first == False and (stride > 1 or is_first):
            self.avg_pool2d_2 = AvgPool2d(
                kernel_size=3, stride=stride,
                data_format='channels_first', padding=1)

        self.conv3 = ConvBNLayer(
            num_channels=group_width,
            num_filters=planes * 4,
            filter_size=1,
            stride=1,
            groups=1,
            act=None,
            name=name + "_conv3")

        if stride != 1 or self.inplanes != self.planes * 4:
            if avg_down:
                if dilation == 1:
                    self.avg_pool2d_3 = AvgPool2d(kernel_size=stride, stride=stride, data_format='channels_first',padding=0)
                else:
                    # self.avg_pool2d_3 = AvgPool2d(kernel_size=1, stride=1, padding=0, ceil_mode=True)
                    self.avg_pool2d_3 = AvgPool2d(kernel_size=1, stride=1, data_format='channels_first', padding=0)
                self.conv4 = GroupConv2d(
                    in_channels=self.inplanes,
                    out_channels=planes * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    n_group=1,
                    data_format='channels_first',
                    # weight_attr=ParamAttr(
                    #     name=name + "_weights", initializer=KaimingNormal()),
                    b_init=None)
            else:
                self.conv4 = GroupConv2d(
                    in_channels=self.inplanes,
                    out_channels=planes * 4,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    n_group=1,
                    data_format='channels_first',
                    # weight_attr=ParamAttr(
                    #     name=name + "_shortcut_weights",
                    #     initializer=KaimingNormal()),
                    b_init=None)

            bn_decay = 0.0
            self._batch_norm = BatchNorm(
                num_features=planes * 4,
                act=None,
                data_format='channels_first',
                # param_attr=ParamAttr(
                #     name=name + "_shortcut_scale",
                #     regularizer=L2Decay(bn_decay)),
                # bias_attr=ParamAttr(
                #     name + "_shortcut_offset", regularizer=L2Decay(bn_decay)),
                # moving_mean_name=name + "_shortcut_mean",
                # moving_variance_name=name + "_shortcut_variance")
                )
    def forward(self, x):
        short = x

        x = self.conv1(x)
        if self.avd and self.avd_first and (self.stride > 1 or self.is_first):
            x = self.avg_pool2d_1(x)

        x = self.conv2(x)

        if self.avd and self.avd_first == False and (self.stride > 1 or
                                                     self.is_first):
            x = self.avg_pool2d_2(x)

        x = self.conv3(x)

        if self.stride != 1 or self.inplanes != self.planes * 4:
            if self.avg_down:
                short = self.avg_pool2d_3(short)

            short = self.conv4(short)

            short = self._batch_norm(short)

        y = tlx.add(short, x)
        y = tlx.relu(y)
        return y


class ResNeStLayer(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 blocks,
                 radix,
                 cardinality,
                 bottleneck_width,
                 avg_down,
                 avd,
                 avd_first,
                 rectify_avg,
                 last_gamma,
                 stride=1,
                 dilation=1,
                 is_first=True,
                 name=None):
        super(ResNeStLayer, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.blocks = blocks
        self.radix = radix
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.avg_down = avg_down
        self.avd = avd
        self.avd_first = avd_first
        self.rectify_avg = rectify_avg
        self.last_gamma = last_gamma
        self.is_first = is_first

        if dilation == 1 or dilation == 2:
            bottleneck_func = self.add_sublayer(
                name + "_bottleneck_0",
                BottleneckBlock(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=stride,
                    radix=radix,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width,
                    avg_down=self.avg_down,
                    avd=avd,
                    avd_first=avd_first,
                    dilation=1,
                    is_first=is_first,
                    rectify_avg=rectify_avg,
                    last_gamma=last_gamma,
                    name=name + "_bottleneck_0"))
        elif dilation == 4:
            bottleneck_func = self.add_sublayer(
                name + "_bottleneck_0",
                BottleneckBlock(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=stride,
                    radix=radix,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width,
                    avg_down=self.avg_down,
                    avd=avd,
                    avd_first=avd_first,
                    dilation=2,
                    is_first=is_first,
                    rectify_avg=rectify_avg,
                    last_gamma=last_gamma,
                    name=name + "_bottleneck_0"))
        else:
            raise RuntimeError("=>unknown dilation size")

        self.inplanes = planes * 4
        self.bottleneck_block_list = [bottleneck_func]
        for i in range(1, blocks):
            curr_name = name + "_bottleneck_" + str(i)

            bottleneck_func = self.add_sublayer(
                curr_name,
                BottleneckBlock(
                    inplanes=self.inplanes,
                    planes=planes,
                    radix=radix,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width,
                    avg_down=self.avg_down,
                    avd=avd,
                    avd_first=avd_first,
                    dilation=dilation,
                    rectify_avg=rectify_avg,
                    last_gamma=last_gamma,
                    name=curr_name))
            self.bottleneck_block_list.append(bottleneck_func)

    def forward(self, x):
        for bottleneck_block in self.bottleneck_block_list:
            x = bottleneck_block(x)
        return x


class ResNeSt(nn.Module):
    def __init__(self,
                 layers,
                 radix=1,
                 groups=1,
                 bottleneck_width=64,
                 dilated=False,
                 dilation=1,
                 deep_stem=False,
                 stem_width=64,
                 avg_down=False,
                 rectify_avg=False,
                 avd=False,
                 avd_first=False,
                 final_drop=0.0,
                 last_gamma=False,
                 class_num=1000):
        super(ResNeSt, self).__init__()

        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        self.deep_stem = deep_stem
        self.stem_width = stem_width
        self.layers = layers
        self.final_drop = final_drop
        self.dilated = dilated
        self.dilation = dilation

        self.rectify_avg = rectify_avg

        if self.deep_stem:
            self.stem = nn.Sequential(OrderedDict([
                ("conv1", ConvBNLayer(
                    num_channels=3,
                    num_filters=stem_width,
                    filter_size=3,
                    stride=2,
                    act="relu",
                    name="conv1")),
                ("conv2", ConvBNLayer(
                        num_channels=stem_width,
                        num_filters=stem_width,
                        filter_size=3,
                        stride=1,
                        act="relu",
                        name="conv2")),
                ("conv3", ConvBNLayer(
                            num_channels=stem_width,
                            num_filters=stem_width * 2,
                            filter_size=3,
                            stride=1,
                            act="relu",
                            name="conv3"))]))
        else:
            self.stem = ConvBNLayer(
                num_channels=3,
                num_filters=stem_width,
                filter_size=7,
                stride=2,
                act="relu",
                name="conv1")

        self.max_pool2d = MaxPool2d(kernel_size=3, stride=2, padding=1, data_format='channels_first')

        self.layer1 = ResNeStLayer(
            inplanes=self.stem_width * 2
            if self.deep_stem else self.stem_width,
            planes=64,
            blocks=self.layers[0],
            radix=radix,
            cardinality=self.cardinality,
            bottleneck_width=bottleneck_width,
            avg_down=self.avg_down,
            avd=avd,
            avd_first=avd_first,
            rectify_avg=rectify_avg,
            last_gamma=last_gamma,
            stride=1,
            dilation=1,
            is_first=False,
            name="layer1")

        #         return

        self.layer2 = ResNeStLayer(
            inplanes=256,
            planes=128,
            blocks=self.layers[1],
            radix=radix,
            cardinality=self.cardinality,
            bottleneck_width=bottleneck_width,
            avg_down=self.avg_down,
            avd=avd,
            avd_first=avd_first,
            rectify_avg=rectify_avg,
            last_gamma=last_gamma,
            stride=2,
            name="layer2")

        if self.dilated or self.dilation == 4:
            self.layer3 = ResNeStLayer(
                inplanes=512,
                planes=256,
                blocks=self.layers[2],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=1,
                dilation=2,
                name="layer3")
            self.layer4 = ResNeStLayer(
                inplanes=1024,
                planes=512,
                blocks=self.layers[3],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=1,
                dilation=4,
                name="layer4")
        elif self.dilation == 2:
            self.layer3 = ResNeStLayer(
                inplanes=512,
                planes=256,
                blocks=self.layers[2],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=2,
                dilation=1,
                name="layer3")
            self.layer4 = ResNeStLayer(
                inplanes=1024,
                planes=512,
                blocks=self.layers[3],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=1,
                dilation=2,
                name="layer4")
        else:
            self.layer3 = ResNeStLayer(
                inplanes=512,
                planes=256,
                blocks=self.layers[2],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=2,
                name="layer3")
            self.layer4 = ResNeStLayer(
                inplanes=1024,
                planes=512,
                blocks=self.layers[3],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=2,
                name="layer4")

        self.pool2d_avg = AdaptiveAvgPool2d(1, data_format='channels_first')

        self.out_channels = 2048

        stdv = 1.0 / math.sqrt(self.out_channels * 1.0)

        self.out = Linear(
            in_features=self.out_channels,
            out_features=class_num,
            # weight_attr=ParamAttr(
            #     initializer=nn.initializer.Uniform(-stdv, stdv),
                # name="fc_weights"),
            # bias_attr=ParamAttr(name="fc_offset")
            )

    def forward(self, x):
        x = self.stem(x)
        x = self.max_pool2d(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)
        x = self.pool2d_avg(x)
        x = tlx.reshape(x, shape=[-1, self.out_channels])
        x = self.out(x)
        # print(x)
        return x

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
#
#     # [print(i,k.shape) for i, k in model.named_parameters()]
#     weights = []
#     for i in range(len(model_state)):
#         model_key = model_state[i]
#         model_key_s, model_key_e = model_key.rsplit('.', 1)
#         if model_key_e in pd2tlx_namelast:
#             new_model_state = model_key_s + '.' + pd2tlx_namelast[model_key_e]
#             weights.append(param[new_model_state])
#         else:
#             print(model_key_e)
#
#     # print(len(model_state), len(param), len(weights))
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


def ResNeSt50_fast_1s1x64d(pretrained=False, use_ssld=False, **kwargs):
    model = ResNeSt(
        layers=[3, 4, 6, 3],
        radix=1,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=True,
        final_drop=0.0,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["ResNeSt50_fast_1s1x64d"],
        use_ssld=use_ssld)
    return model


def ResNeSt50(pretrained=False, use_ssld=False, **kwargs):
    model = ResNeSt(
        layers=[3, 4, 6, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=False,
        final_drop=0.0,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ResNeSt50"], use_ssld=use_ssld)
    return model


def ResNeSt101(pretrained=False, use_ssld=False, **kwargs):
    model = ResNeSt(
        layers=[3, 4, 23, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        final_drop=0.0,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ResNeSt101"], use_ssld=use_ssld)
    return model

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


    model = ResNeSt50(pretrained=True)
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