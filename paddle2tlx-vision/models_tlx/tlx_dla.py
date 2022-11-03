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

# Code was based on https://github.com/ucbdrive/dla
# reference: https://arxiv.org/abs/1707.06484

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorlayerx.files import assign_weights
import os
os.environ['TL_BACKEND'] = 'paddle'
import paddle
import math
# import paddle.nn as nn
import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from utils.download import get_weights_path_from_url
from utils.load_model import restore_model
# from paddle.nn import Identity
from paddle.nn.initializer import Normal, Constant
model_urls = {
    "DLA34":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA34_pretrained.pdparams",
    "DLA102":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA102_pretrained.pdparams",
}

__all__ = []

zeros_ = tlx.initializers.constant(value=0.)
ones_ = tlx.initializers.constant(value=1.)

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class DlaBasic(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, **cargs):
        super(DlaBasic, self).__init__()
        self.conv1 = nn.Conv2d(out_channels=planes, kernel_size=(3, 3), stride=(stride, stride),dilation=dilation,b_init=None, act=None, padding=dilation,
                               in_channels=inplanes, data_format='channels_first')
        self.bn1 = nn.BatchNorm2d(num_features=planes, data_format='channels_first')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels=planes, kernel_size=(3, 3), stride=(1, 1),dilation=dilation,b_init=None, act=None, padding=dilation,
                               in_channels=planes, data_format='channels_first')
        self.bn2 = nn.BatchNorm2d(num_features=planes, data_format='channels_first')
        self.stride = stride
    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)

        return out


class DlaBottleneck(nn.Module):
    expansion = 2

    def __init__(self,
                 inplanes,
                 outplanes,
                 stride=1,
                 dilation=1,
                 cardinality=1,
                 base_width=64):
        super(DlaBottleneck, self).__init__()
        self.stride = stride
        mid_planes = int(
            math.floor(outplanes * (base_width / 64)) * cardinality)
        mid_planes = mid_planes // self.expansion
        self.conv1 = nn.Conv2d(out_channels=mid_planes, kernel_size=(1, 1), stride=(1, 1),b_init=None, act=None, padding=0,
                               in_channels=inplanes, data_format='channels_first')
        self.bn1 = nn.BatchNorm2d(num_features=mid_planes, data_format='channels_first')
        self.conv2 = nn.Conv2d(out_channels=mid_planes, kernel_size=(3, 3), stride=(stride, stride),dilation=dilation,b_init=None, act=None, padding=dilation,
                               in_channels=mid_planes, data_format='channels_first')
        self.bn2 = nn.BatchNorm2d(num_features=mid_planes, data_format='channels_first')
        self.conv3 = nn.Conv2d(out_channels=outplanes, kernel_size=(1, 1), stride=(1, 1),b_init=None, act=None, padding=0,
                               in_channels=mid_planes, data_format='channels_first')
        self.bn3 = nn.BatchNorm2d(num_features=outplanes, data_format='channels_first')
        self.relu = nn.ReLU()

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class DlaRoot(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(DlaRoot, self).__init__()
        self.conv = nn.Conv2d(out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1),b_init=None, act=None,
                              padding=(kernel_size - 1) // 2,
                              in_channels=in_channels, data_format='channels_first')
        self.bn = nn.BatchNorm2d(num_features=out_channels, data_format='channels_first')
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(tlx.concat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class DlaTree(nn.Module):
    def __init__(self,
                 levels,
                 block,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 cardinality=1,
                 base_width=64,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 root_residual=False):
        super(DlaTree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels

        self.downsample = nn.MaxPool2d(kernel_size=stride, stride=stride, padding=0, data_format='channels_first') if stride > 1 else Identity()
        self.project = Identity()
        cargs = dict(
            dilation=dilation, cardinality=cardinality, base_width=base_width)

        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, **cargs)
            self.tree2 = block(out_channels, out_channels, 1, **cargs)
            if in_channels != out_channels:
                self.project = nn.Sequential(
                    nn.Conv2d(out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1),b_init=None, act=None,
                              in_channels=in_channels, data_format='channels_first')
                    ,
                    nn.BatchNorm2d(num_features=out_channels, data_format='channels_first'))
        else:
            cargs.update(
                dict(
                    root_kernel_size=root_kernel_size,
                    root_residual=root_residual))
            self.tree1 = DlaTree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                **cargs)
            self.tree2 = DlaTree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                **cargs)

        if levels == 1:
            self.root = DlaRoot(root_dim, out_channels, root_kernel_size,
                                root_residual)

        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x)
        residual = self.project(bottom)

        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)

        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self,
                 levels,
                 channels,
                 in_chans=3,
                 cardinality=1,
                 base_width=64,
                 block=DlaBottleneck,
                 residual_root=False,
                 drop_rate=0.0,
                 class_num=1000,
                 with_pool=True):
        super(DLA, self).__init__()
        self.channels = channels
        self.class_num = class_num
        self.with_pool = with_pool
        self.cardinality = cardinality
        self.base_width = base_width
        self.drop_rate = drop_rate

        self.base_layer = nn.Sequential(
            nn.Conv2d(out_channels=channels[0], kernel_size=(7, 7), stride=(1, 1), b_init=None,act=None, padding=3,
                  in_channels=in_chans, data_format='channels_first')
            ,
            nn.BatchNorm2d(num_features=channels[0], data_format='channels_first'),
            nn.ReLU())
        self.level0 = self._make_conv_level(channels[0], channels[0],
                                            levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)

        cargs = dict(
            cardinality=cardinality,
            base_width=base_width,
            root_residual=residual_root)

        self.level2 = DlaTree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            **cargs)
        self.level3 = DlaTree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            **cargs)
        self.level4 = DlaTree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            **cargs)
        self.level5 = DlaTree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            **cargs)

        self.feature_info = [
            # rare to have a meaningful stride 1 level
            dict(
                num_chs=channels[0], reduction=1, module='level0'),
            dict(
                num_chs=channels[1], reduction=2, module='level1'),
            dict(
                num_chs=channels[2], reduction=4, module='level2'),
            dict(
                num_chs=channels[3], reduction=8, module='level3'),
            dict(
                num_chs=channels[4], reduction=16, module='level4'),
            dict(
                num_chs=channels[5], reduction=32, module='level5'),
        ]

        self.num_features = channels[-1]

        if with_pool:
            self.global_pool = nn.AdaptiveAvgPool2d(1, data_format='channels_first')

        if class_num > 0:
            self.fc = nn.Conv2d(out_channels=class_num, kernel_size=(1, 1), stride=(1, 1), act=None,padding=0,
                                in_channels=self.num_features, data_format='channels_first')

        for m in self.sublayers():

            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                normal_ = tlx.initializers.random_normal(mean=0.0, stddev=math.sqrt(2. / n))
                normal_(m.filters)
            elif isinstance(m, nn.BatchNorm2d):
                ones_(m.gamma)
                zeros_(m.beta)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(out_channels=planes, kernel_size=(3, 3), stride=(stride, stride) if i == 0 else (1, 1),
                          dilation=dilation,b_init=None,act=None,padding=dilation, in_channels=inplanes, data_format='channels_first'),
                nn.BatchNorm2d(num_features=planes, data_format='channels_first'),
                nn.ReLU()])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward_features(self, x):
        x = self.base_layer(x)

        x = self.level0(x)
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)
        x = self.level5(x)

        return x

    def forward(self, x):

        x = self.forward_features(x)

        if self.with_pool:
            x = self.global_pool(x)

        if self.drop_rate > 0.:
            x = nn.Dropout(x, p=self.drop_rate, training=self.training)

        if self.class_num > 0:

            x = self.fc(x)

            x = nn.Flatten()(x)


        return x


def re_model(param, model):
    tlx2pd_namelast = {'filters': 'weight',        # conv2d
                       'biases': 'bias',           # linear
                       'weights': 'weight',        # linear
                       'gamma': 'weight',          # bn
                       'beta': 'bias',             # bn
                       'moving_mean': '_mean',     # bn
                       'moving_var': '_variance',  # bn
                       }

    model_state = [i for i, k in model.named_parameters()]
    weights = []
    for i in range(len(model_state)):
        model_key = model_state[i]
        key_list = model_key.split('.')
        model_key_e = key_list[-1]
        if model_key_e in tlx2pd_namelast:
            new_model_state = '.'.join(key_list[:len(key_list)-1]) +'.' + tlx2pd_namelast[model_key_e]
            weights.append(param[new_model_state])
        else:
            print(model_key_e)

    assign_weights(weights, model)
    del weights


def _dla(arch, pretrained, **kwargs):

    if arch == "DLA34":
        model = DLA(levels=(1, 1, 1, 2, 2, 1),
                    channels=(16, 32, 64, 128, 256, 512),
                    block=DlaBasic,
                    **kwargs)
    else:
        model = DLA(levels=(1, 1, 1, 3, 4, 1),
                    channels=(16, 32, 128, 256, 512, 1024),
                    block=DlaBottleneck,
                    residual_root=True,
                    **kwargs)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch])

        param = paddle.load(weight_path)
        re_model(param, model)



    return model

def dla34(pretrained=False, **kwargs):

    return _dla("DLA34",pretrained, **kwargs)

def dla102(pretrained=False, **kwargs):

    return _dla("DLA102",pretrained, **kwargs)


