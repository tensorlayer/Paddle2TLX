# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
#
# Code was heavily based on https://github.com/facebookresearch/ConvNeXt

import paddle
import paddle.nn as nn
from paddle.nn import Identity
from paddle.nn.initializer import TruncatedNormal, Constant
# from ops.tlx_activation import tlx_GELU  # todo
from paddle2tlx.pd2tlx.utils import load_model_clas

MODEL_URLS = {
    "ConvNeXt_tiny": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ConvNeXt_tiny_pretrained.pdparams",  # TODO
}

__all__ = list(MODEL_URLS.keys())

trunc_normal_ = TruncatedNormal(std=.02)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    s = paddle.shape(x)
    shape = (s[0], ) + (1, ) * (x.ndim - 1)
    r = paddle.rand(shape, dtype=x.dtype)
    random_tensor = keep_prob + r
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        try:
            self.training = self.is_train
        except:
            pass
        return drop_path(x, self.drop_prob, self.training)


# class Identity(nn.Layer):
#     def __init__(self):
#         super(Identity, self).__init__()
#
#     def forward(self, input):
#         return input


class ChannelsFirstLayerNorm(nn.Layer):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, epsilon=1e-5):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[normalized_shape], default_initializer=ones_)
        self.bias = self.create_parameter(
            shape=[normalized_shape], default_initializer=zeros_)
        self.epsilon = epsilon
        self.normalized_shape = [normalized_shape]

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        sq = paddle.sqrt(s + self.epsilon)
        x = (x - u) / sq
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Block(nn.Layer):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2D(
            dim, dim, 7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, epsilon=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        if layer_scale_init_value > 0:
            self.gamma = self.create_parameter(
                shape=[dim],
                default_initializer=Constant(value=layer_scale_init_value))
        else:
            self.gamma = None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.transpose([0, 2, 3, 1])  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose([0, 3, 1, 2])  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Layer):
    r""" ConvNeXt
        A PaddlePaddle impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        class_num (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self,
                 in_chans=3,
                 class_num=1000,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.):
        super().__init__()

        # stem and 3 intermediate downsampling conv layers
        self.downsample_layers = nn.LayerList()
        stem = nn.Sequential(
            nn.Conv2D(
                in_chans, dims[0], 4, stride=4),
            ChannelsFirstLayerNorm(
                dims[0], epsilon=1e-6))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                ChannelsFirstLayerNorm(
                    dims[i], epsilon=1e-6),
                nn.Conv2D(
                    dims[i], dims[i + 1], 2, stride=2), )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.LayerList()
        dp_rates = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(* [
                Block(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], epsilon=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], class_num)

        self.apply(self._init_weights)
        try:
            self.head.weight.set_value(self.head.weight * head_init_scale)
            self.head.bias.set_value(self.head.bias * head_init_scale)
        except:
            self.head.weights.set_value(self.head.weights * head_init_scale)
            self.head.biases.set_value(self.head.biases * head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            try:
                trunc_normal_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            except:
                trunc_normal_(m.filters)
                if m.biases is not None:
                    zeros_(m.biases)
        if isinstance(m, nn.Linear):
            try:
                trunc_normal_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            except:
                trunc_normal_(m.weights)
                if m.biases is not None:
                    zeros_(m.biases)
    
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # global average pooling, (N, C, H, W) -> (N, C)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# def _load_pretrained(pretrained, model, model_url):
#     if pretrained is False:
#         pass
#     elif pretrained is True:
#         weight_path = get_weights_path_from_url(model_url)
#         param = paddle.load(weight_path)
#         # model_state = [i for i, kk in model.named_parameters()]
#         # param_state = [i for i, k in param.items()]
#         model.set_dict(param)
#
#
# def ConvNeXt_tiny(pretrained=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
#     _load_pretrained(pretrained, model, MODEL_URLS["ConvNeXt_tiny"])
#     return model
#
# def convnext(pretrained=False, **kwargs):
#     model = ConvNeXt_tiny(pretrained=pretrained, **kwargs)
#     return model


def _convnext(arch, pretrained, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

    if pretrained:
        model = load_model_clas(model, arch, MODEL_URLS)

    return model


def convnext(pretrained=False, **kwargs):
    """AlexNet model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            from paddle.vision.models import alexnet

            # build model
            model = alexnet()

            # build model and load imagenet pretrained weight
            # model = alexnet(pretrained=True)
    """
    return _convnext('ConvNeXt_tiny', pretrained, **kwargs)