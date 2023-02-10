# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
# from models import utils
from .layers import ConvBNReLU
# from utils.download import download_and_decompress
# from utils.load_model import load_model_seg, restore_model_seg
# from ops.tlx_extend import get_tensor_shape  # todo
# from ops.tlx_basic_pooling import tlx_MaxPool2d
# from ops.tlx_basic_conv import tlx_ConvTranspose2d
from paddle2tlx.pd2tlx.utils import load_model_rsseg

UNET_URLS = "https://bj.bcebos.com/paddleseg/dygraph/cityscapes/unet_cityscapes_1024x512_160k/model.pdparams"


class UNet(nn.Layer):
    """
    The UNet implementation based on PaddlePaddle.

    The original article refers to
    Olaf Ronneberger, et, al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (https://arxiv.org/abs/1505.04597).

    Args:
        num_classes (int): The unique number of target classes.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        use_deconv (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
        in_channels (int, optional): The channels of input image. Default: 3.
        pretrained (str, optional): The path or url of pretrained model for fine tuning. Default: None.
    """

    def __init__(self,
                 num_classes,
                 align_corners=False,
                 use_deconv=False,
                 in_channels=3,
                 pretrained=None):
        super().__init__()

        self.encode = Encoder(in_channels)
        self.decode = Decoder(align_corners, use_deconv=use_deconv)
        self.cls = self.conv = nn.Conv2D(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

        self.pretrained = pretrained
        # self.init_weight()  # comment

    def forward(self, x):
        logit_list = []
        x, short_cuts = self.encode(x)
        # print(f"unet_pd encode.x={x}")
        # print(f"unet input short_cuts={short_cuts}")
        x = self.decode(x, short_cuts)
        logit = self.cls(x)
        # print(f"logit x={logit}")
        logit_list.append(logit)
        return logit_list

    # def init_weight(self):
    #     if self.pretrained is not None:
    #         utils.load_entire_model(self, self.pretrained)


class Encoder(nn.Layer):
    def __init__(self, in_channels=3):
        super().__init__()

        self.double_conv = nn.Sequential(
            ConvBNReLU(in_channels, 64, 3), ConvBNReLU(64, 64, 3))
        down_channels = [[64, 128], [128, 256], [256, 512], [512, 512]]
        self.down_sample_list = nn.LayerList([
            self.down_sampling(channel[0], channel[1])
            for channel in down_channels
        ])

    def down_sampling(self, in_channels, out_channels):
        modules = []
        modules.append(nn.MaxPool2D(kernel_size=2, stride=2))
        modules.append(ConvBNReLU(in_channels, out_channels, 3))
        modules.append(ConvBNReLU(out_channels, out_channels, 3))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        # print(f"unet before double_conv.x={x}")
        x = self.double_conv(x)
        # print(f"unet after double_conv.x={x}") # [1, 64, 512, 512]
        # print("===================================")
        # print(f"unet_pd after double_conv.x.shape={x.shape}")
        # i = 0
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
            # print(f"paddle, down_sample {i}, x.shape={x.shape}")
            # print(f"paddle, down_sample {i}, x.shape={x}")
            # i +=1
        return x, short_cuts


class Decoder(nn.Layer):
    def __init__(self, align_corners, use_deconv=False):
        super().__init__()

        up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        self.up_sample_list = nn.LayerList([
            UpSampling(channel[0], channel[1], align_corners, use_deconv)
            for channel in up_channels
        ])

    def forward(self, x, short_cuts):
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
        return x


class UpSampling(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_deconv=False):
        super().__init__()

        self.align_corners = align_corners

        self.use_deconv = use_deconv
        if self.use_deconv:
            self.deconv = nn.Conv2DTranspose(
                in_channels,
                out_channels // 2,
                kernel_size=2,
                stride=2,
                padding=0)
            in_channels = in_channels + out_channels // 2
        else:
            in_channels *= 2

        self.double_conv = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3),
            ConvBNReLU(out_channels, out_channels, 3))

    def forward(self, x, short_cut):
        if self.use_deconv:
            x = self.deconv(x)
        else:
            x = F.interpolate(
                x,
                paddle.shape(short_cut)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x


def _unet(pretrained=None, num_classes=19):
    model = UNet(num_classes=num_classes)
    if pretrained:
        model = load_model_rsseg(model, "unet", UNET_URLS)
        # weight_path = download_and_decompress(UNET_URLS, "unet")
        # param = paddle.load(weight_path)
        # model.load_dict(param)
    return model
