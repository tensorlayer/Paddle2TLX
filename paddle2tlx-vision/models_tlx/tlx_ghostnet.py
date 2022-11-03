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

# Code was based on https://github.com/huawei-noah/CV-Backbones/tree/master/ghostnet_pytorch
# reference: https://arxiv.org/abs/1911.11907
import os

os.environ['TL_BACKEND'] = 'paddle'
import tensorlayerx
import math
import paddle
# from paddle import ParamAttr
# import paddle.nn as nn
import paddle.nn.functional as F
# from paddle.nn import Conv2D, BatchNorm, AdaptiveAvgPool2D, Linear
# from paddle.regularizer import L2Decay
# from paddle.nn.initializer import Uniform, KaimingNormal
from utils.download import get_weights_path_from_url

from utils.load_model import restore_model

MODEL_URLS = {
    "GhostNet_x0_5":
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x0_5_pretrained.pdparams",
    "GhostNet_x1_0":
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_0_pretrained.pdparams",
    "GhostNet_x1_3":
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_3_pretrained.pdparams",
}

__all__ = []


class ConvBNLayer(tensorlayerx.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 act="relu",
                 name=None):
        super(ConvBNLayer, self).__init__()
        self._conv = tensorlayerx.nn.GroupConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            n_group=groups,
            # weight_attr=ParamAttr(
            #     initializer=KaimingNormal(), name=name + "_weights"),
            # bias_attr=False
            b_init=None,
            data_format='channels_first',
        )
        # self._conv = Conv2D(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=(kernel_size - 1) // 2,
        #     groups=groups,
        #     weight_attr=ParamAttr(
        #         initializer=KaimingNormal(), name=name + "_weights"),
        #     bias_attr=False)

        bn_name = name + "_bn"

        self._batch_norm = tensorlayerx.nn.BatchNorm(
            num_features=out_channels,
            act=act,
            data_format='channels_first',
            # param_attr=ParamAttr(
            #     name=bn_name + "_scale", regularizer=L2Decay(0.0)),
            # bias_attr=ParamAttr(
            #     name=bn_name + "_offset", regularizer=L2Decay(0.0)),
            # moving_mean_name=bn_name + "_mean",
            # moving_variance_name=bn_name + "_variance"
        )
        # self._batch_norm = BatchNorm(
        #     num_channels=out_channels,
        #     act=act,
        #     param_attr=ParamAttr(
        #         name=bn_name + "_scale", regularizer=L2Decay(0.0)),
        #     bias_attr=ParamAttr(
        #         name=bn_name + "_offset", regularizer=L2Decay(0.0)),
        #     moving_mean_name=bn_name + "_mean",
        #     moving_variance_name=bn_name + "_variance")

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class SEBlock(tensorlayerx.nn.Module):
    def __init__(self, num_channels, reduction_ratio=4, name=None):
        super(SEBlock, self).__init__()
        self.pool2d_gap = tensorlayerx.nn.AdaptiveAvgPool2d(1, data_format='channels_first', )
        # self.pool2d_gap = AdaptiveAvgPool2D(1)
        self._num_channels = num_channels
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        med_ch = num_channels // reduction_ratio
        self.squeeze = tensorlayerx.nn.Linear(
            in_features=num_channels,
            out_features=med_ch,
            # weight_attr=ParamAttr(
            #     initializer=Uniform(-stdv, stdv), name=name + "_1_weights"),
            # bias_attr=ParamAttr(name=name + "_1_offset")
        )
        # self.squeeze = Linear(
        #     num_channels,
        #     med_ch,
        #     weight_attr=ParamAttr(
        #         initializer=Uniform(-stdv, stdv), name=name + "_1_weights"),
        #     bias_attr=ParamAttr(name=name + "_1_offset"))

        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = tensorlayerx.nn.Linear(
            in_features=med_ch,
            out_features=num_channels,
            # weight_attr=ParamAttr(
            #     initializer=Uniform(-stdv, stdv), name=name + "_2_weights"),
            # bias_attr=ParamAttr(name=name + "_2_offset")
        )
        # self.excitation = Linear(
        #     med_ch,
        #     num_channels,
        #     weight_attr=ParamAttr(
        #         initializer=Uniform(-stdv, stdv), name=name + "_2_weights"),
        #     bias_attr=ParamAttr(name=name + "_2_offset"))

    def forward(self, inputs):
        pool = self.pool2d_gap(inputs)
        pool = tensorlayerx.squeeze(pool, axis=[2, 3])
        # pool = paddle.squeeze(pool, axis=[2, 3])
        squeeze = self.squeeze(pool)
        squeeze = F.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = tensorlayerx.ops.clip_by_value(t=excitation, clip_value_min=0, clip_value_max=1)
        # excitation = paddle.clip(x=excitation, min=0, max=1)

        excitation = tensorlayerx.ops.expand_dims(input=excitation, axis=[2, 3])
        # excitation = paddle.unsqueeze(excitation, axis=[2, 3])

        out = tensorlayerx.multiply(inputs, excitation)
        # out = paddle.multiply(inputs, excitation)
        return out


class GhostModule(tensorlayerx.nn.Module):
    def __init__(self,
                 in_channels,
                 output_channels,
                 kernel_size=1,
                 ratio=2,
                 dw_size=3,
                 stride=1,
                 relu=True,
                 name=None):
        super(GhostModule, self).__init__()
        init_channels = int(math.ceil(output_channels / ratio))
        new_channels = int(init_channels * (ratio - 1))
        self.primary_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=init_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=1,
            act="relu" if relu else None,
            name=name + "_primary_conv")
        self.cheap_operation = ConvBNLayer(
            in_channels=init_channels,
            out_channels=new_channels,
            kernel_size=dw_size,
            stride=1,
            groups=init_channels,
            act="relu" if relu else None,
            name=name + "_cheap_operation")

    def forward(self, inputs):
        x = self.primary_conv(inputs)
        y = self.cheap_operation(x)
        out = tensorlayerx.concat([x, y], axis=1)
        # out = paddle.concat([x, y], axis=1)
        return out


class GhostBottleneck(tensorlayerx.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 output_channels,
                 kernel_size,
                 stride,
                 use_se,
                 name=None):
        super(GhostBottleneck, self).__init__()
        self._stride = stride
        self._use_se = use_se
        self._num_channels = in_channels
        self._output_channels = output_channels
        self.ghost_module_1 = GhostModule(
            in_channels=in_channels,
            output_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            relu=True,
            name=name + "_ghost_module_1")
        if stride == 2:
            self.depthwise_conv = ConvBNLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                groups=hidden_dim,
                act=None,
                name=name +
                     "_depthwise_depthwise"  # looks strange due to an old typo, will be fixed later.
            )
        if use_se:
            self.se_block = SEBlock(num_channels=hidden_dim, name=name + "_se")
        self.ghost_module_2 = GhostModule(
            in_channels=hidden_dim,
            output_channels=output_channels,
            kernel_size=1,
            relu=False,
            name=name + "_ghost_module_2")
        if stride != 1 or in_channels != output_channels:
            self.shortcut_depthwise = ConvBNLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=in_channels,
                act=None,
                name=name +
                     "_shortcut_depthwise_depthwise"  # looks strange due to an old typo, will be fixed later.
            )
            self.shortcut_conv = ConvBNLayer(
                in_channels=in_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                act=None,
                name=name + "_shortcut_conv")

    def forward(self, inputs):
        x = self.ghost_module_1(inputs)
        if self._stride == 2:
            x = self.depthwise_conv(x)
        if self._use_se:
            x = self.se_block(x)
        x = self.ghost_module_2(x)
        if self._stride == 1 and self._num_channels == self._output_channels:
            shortcut = inputs
        else:
            shortcut = self.shortcut_depthwise(inputs)
            shortcut = self.shortcut_conv(shortcut)
        return tensorlayerx.ops.add(x, shortcut)
        # return paddle.add(x=x, y=shortcut)


class GhostNet(tensorlayerx.nn.Module):
    def __init__(self, scale, class_num=1000):
        super(GhostNet, self).__init__()
        self.cfgs = [
            # k, t, c, SE, s
            [3, 16, 16, 0, 1],
            [3, 48, 24, 0, 2],
            [3, 72, 24, 0, 1],
            [5, 72, 40, 1, 2],
            [5, 120, 40, 1, 1],
            [3, 240, 80, 0, 2],
            [3, 200, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 480, 112, 1, 1],
            [3, 672, 112, 1, 1],
            [5, 672, 160, 1, 2],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1]
        ]
        self.scale = scale
        output_channels = int(self._make_divisible(16 * self.scale, 4))
        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=output_channels,
            kernel_size=3,
            stride=2,
            groups=1,
            act="relu",
            name="conv1")
        # build inverted residual blocks
        idx = 0
        self.ghost_bottleneck_list = []
        for k, exp_size, c, use_se, s in self.cfgs:
            in_channels = output_channels
            output_channels = int(self._make_divisible(c * self.scale, 4))
            hidden_dim = int(self._make_divisible(exp_size * self.scale, 4))
            ghost_bottleneck = self.add_sublayer(
                name="_ghostbottleneck_" + str(idx),
                sublayer=GhostBottleneck(
                    in_channels=in_channels,
                    hidden_dim=hidden_dim,
                    output_channels=output_channels,
                    kernel_size=k,
                    stride=s,
                    use_se=use_se,
                    name="_ghostbottleneck_" + str(idx)))
            self.ghost_bottleneck_list.append(ghost_bottleneck)
            idx += 1
        # build last several layers
        in_channels = output_channels
        output_channels = int(self._make_divisible(exp_size * self.scale, 4))
        self.conv_last = ConvBNLayer(
            in_channels=in_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act="relu",
            name="conv_last")
        self.pool2d_gap = tensorlayerx.nn.AdaptiveAvgPool2d(1, data_format='channels_first', )
        # self.pool2d_gap = AdaptiveAvgPool2D(1)
        in_channels = output_channels
        self._fc0_output_channels = 1280
        self.fc_0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=self._fc0_output_channels,
            kernel_size=1,
            stride=1,
            act="relu",
            name="fc_0")
        self.dropout = tensorlayerx.nn.Dropout(p=0.2)
        stdv = 1.0 / math.sqrt(self._fc0_output_channels * 1.0)
        self.fc_1 = tensorlayerx.nn.Linear(
            in_features=self._fc0_output_channels,
            out_features=class_num,
            # weight_attr=ParamAttr(
            #     name="fc_1_weights", initializer=Uniform(-stdv, stdv)),
            # bias_attr=ParamAttr(name="fc_1_offset")
        )
        # self.fc_1 = Linear(
        #     self._fc0_output_channels,
        #     class_num,
        #     weight_attr=ParamAttr(
        #         name="fc_1_weights", initializer=Uniform(-stdv, stdv)),
        #     bias_attr=ParamAttr(name="fc_1_offset"))

    def forward(self, inputs):
        x = self.conv1(inputs)
        for ghost_bottleneck in self.ghost_bottleneck_list:
            x = ghost_bottleneck(x)
        x = self.conv_last(x)
        x = self.pool2d_gap(x)
        x = self.fc_0(x)
        x = self.dropout(x)
        x = tensorlayerx.reshape(x, shape=[-1, self._fc0_output_channels])
        # x = paddle.reshape(x, shape=[-1, self._fc0_output_channels])
        x = self.fc_1(x)
        return x

    def _make_divisible(self, v, divisor, min_value=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


# def restore_model(param, model):
#     from tensorlayerx.files import assign_weights
#     tlx2pd_namelast = {'filters': 'weight',  # conv2d
#                        'biases': 'bias',  # linear
#                        'weights': 'weight',  # linear
#                        'gamma': 'weight',  # bn
#                        'beta': 'bias',  # bn
#                        'moving_mean': '_mean',  # bn
#                        'moving_var': '_variance',  # bn
#                        }
#     # print([{i: k} for i, k in model.named_parameters()])
#     model_state = [i for i, k in model.named_parameters()]
#     # for i, k in model.named_parameters():
#     #     print(i)
#     # exit()
#     weights = []
#
#     for i in range(len(model_state)):
#         model_key = model_state[i]
#         model_key_s, model_key_e = model_key.rsplit('.', 1)
#         # print(model_key_s, model_key_e)
#         if model_key_e in tlx2pd_namelast:
#             new_model_state = model_key_s + '.' + tlx2pd_namelast[model_key_e]
#             weights.append(param[new_model_state])
#         else:
#             print('**' * 10, model_key)
#     assign_weights(weights, model)
#     del weights


def ghostnet(pretrained=False, **kwargs):
    model = GhostNet(scale=0.5, **kwargs)
    # print(paddle.summary(model, [(1, 3, 256, 256)]))
    # a = 0
    # for i, j in model.named_parameters():
    #     print(str(a), '-' * 10, i)
    # exit()
    if pretrained:
        assert 'GhostNet_x0_5' in MODEL_URLS, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            'GhostNet_x0_5')
        weight_path = get_weights_path_from_url(MODEL_URLS['GhostNet_x0_5'])
        param = paddle.load(weight_path)
        # a = 0
        # for i, j in param.items():
        #     print(str(a), '-' * 10, i)
        # exit()
        restore_model(param, model)
    return model


if __name__ == '__main__':
    from PIL import Image
    from paddle import to_tensor
    import tensorlayerx as tlx
    import numpy as np


    def load_image(image_path, mode=None):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = np.array(img).astype(np.float32)
        if mode == "tlx":
            img = np.expand_dims(img, 0)
            img = img / 255.0
            img = tlx.convert_to_tensor(img)
            img = tlx.ops.nhwc_to_nchw(img)
        elif mode == "pd":
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, 0)
            img = img / 255.0
            img = to_tensor(img)
        elif mode == "pt":
            pass
        else:
            img = img.transpose((2, 0, 1))
            img = img / 255.0
        return img


    image_file = "../../images/dog.jpeg"
    model_tlx = ghostnet(pretrained=True)
    model_tlx.set_eval()
    img = load_image(image_file, "tlx")
    result_tlx = model_tlx(img)

    file_path = '../../images/imagenet_classes.txt'
    with open(file_path) as f:
        classes = [line.strip() for line in f.readlines()]
    print(classes[np.argmax(result_tlx[0])])
