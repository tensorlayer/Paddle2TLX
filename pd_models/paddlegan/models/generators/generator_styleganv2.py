#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# code was heavily based on https://github.com/rosinality/stylegan2-pytorch
# MIT License
# Copyright (c) 2019 Kim Seonghyeon

import math
import random
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .builder import GENERATORS
from ..layers.equalized import EqualLinear
from ..layers.fused_act import FusedLeakyReLU
from ..layers.upfirdn2d import Upfirdn2dUpsample, Upfirdn2dBlur


class PixelNorm(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs * paddle.rsqrt(
            paddle.mean(inputs * inputs, 1, keepdim=True) + 1e-8)


class ModulatedConv2D(nn.Layer):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Upfirdn2dBlur(blur_kernel,
                                      pad=(pad0, pad1),
                                      upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Upfirdn2dBlur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * (kernel_size * kernel_size)
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        w_init = nn.initializer.Normal()
        self.weight = self.create_parameter(
            (1, out_channel, in_channel, kernel_size, kernel_size),
            default_initializer=w_init)

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})")

    def forward(self, inputs, style, apply_modulation=False):
        batch, in_channel, height, width = inputs.shape

        if apply_modulation: style = self.modulation(style)
        style = style.reshape((batch, 1, in_channel, 1, 1))
        weight = self.scale * self.weight * style
        del style

        if self.demodulate:
            demod = paddle.rsqrt((weight * weight).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.reshape((batch, self.out_channel, 1, 1, 1))

        weight = weight.reshape((batch * self.out_channel, in_channel,
                                 self.kernel_size, self.kernel_size))

        if self.upsample:
            inputs = inputs.reshape((1, batch * in_channel, height, width))
            weight = weight.reshape((batch, self.out_channel, in_channel,
                                     self.kernel_size, self.kernel_size))
            weight = weight.transpose((0, 2, 1, 3, 4)).reshape(
                (batch * in_channel, self.out_channel, self.kernel_size,
                 self.kernel_size))
            out = F.conv2d_transpose(inputs,
                                     weight,
                                     padding=0,
                                     stride=2,
                                     groups=batch)
            _, _, height, width = out.shape
            out = out.reshape((batch, self.out_channel, height, width))
            out = self.blur(out)

        elif self.downsample:
            inputs = self.blur(inputs)
            _, _, height, width = inputs.shape
            inputs = inputs.reshape((1, batch * in_channel, height, width))
            out = F.conv2d(inputs, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.reshape((batch, self.out_channel, height, width))

        else:
            inputs = inputs.reshape((1, batch * in_channel, height, width))
            out = F.conv2d(inputs, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.reshape((batch, self.out_channel, height, width))

        return out


class NoiseInjection(nn.Layer):
    def __init__(self, is_concat=False):
        super().__init__()
        _init = nn.initializer.Constant(0.0)
        # self.weight = self.create_parameter(
        #     (1,), default_initializer=_init)
        self.weight = self.create_parameter([1, ], default_initializer=_init)
        self.is_concat = is_concat

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = paddle.randn((batch, 1, height, width))
        if self.is_concat:
            return paddle.concat([image, self.weight * noise], axis=1)
        else:
            return image + self.weight * noise


class ConstantInput(nn.Layer):
    def __init__(self, channel, size=4):
        super().__init__()
        _init = nn.initializer.Normal()
        self.input = self.create_parameter(
            (1, channel, size, size),
            default_initializer=_init)

    def forward(self, batch):
        out = self.input.tile((batch, 1, 1, 1))

        return out


class StyledConv(nn.Layer):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 style_dim,
                 upsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 demodulate=True,
                 is_concat=False):
        super().__init__()

        self.conv = ModulatedConv2D(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection(is_concat=is_concat)
        self.activate = FusedLeakyReLU(out_channel *
                                       2 if is_concat else out_channel)

    def forward(self, inputs, style, noise=None):
        out = self.conv(inputs, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class ToRGB(nn.Layer):
    def __init__(self,
                 in_channel,
                 style_dim,
                 upsample=True,
                 blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upfirdn2dUpsample(blur_kernel)

        self.conv = ModulatedConv2D(in_channel,
                                    3,
                                    1,
                                    style_dim,
                                    demodulate=False)
        _init = nn.initializer.Constant(0.0)
        self.bias = self.create_parameter((1, 3, 1, 1),
                                          default_initializer=_init)

    def forward(self, inputs, style, skip=None):
        out = self.conv(inputs, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


@GENERATORS.register()
class StyleGANv2Generator(nn.Layer):
    def __init__(self,
                 size,
                 style_dim,
                 n_mlp,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 is_concat=False):
        super().__init__()

        self.size = size
        self.style_dim = style_dim
        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(style_dim,
                            style_dim,
                            lr_mul=lr_mlp,
                            activation="fused_lrelu"))

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        self.channels_lst = []
        self.w_idx_lst = [
            0,1,        # 4
            1,2,3,      # 8
            3,4,5,      # 16
            5,6,7,      # 32
            7,8,9,      # 64
            9,10,11,    # 128
            11,12,13,   # 256
            13,14,15,   # 512
            15,16,17,   # 1024
        ]
        self.style_layers = [
            0,    #1,
            2, 3, #4,
            5, 6, #7,
            8, 9, #10,
            11, 12,# 13,
            14, 15,# 16,
            17, 18,# 19,
            20, 21,# 22,
            23, 24,# 25
        ]

        if self.log_size != 10:
            self.w_idx_lst = self.w_idx_lst[:-(3 * (10 - self.log_size))]
            self.style_layers = self.style_layers[:-(2 * (10 - self.log_size))]

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4],
                                self.channels[4],
                                3,
                                style_dim,
                                blur_kernel=blur_kernel,
                                is_concat=is_concat)
        self.to_rgb1 = ToRGB(self.channels[4] *
                             2 if is_concat else self.channels[4],
                             style_dim,
                             upsample=False)
        self.channels_lst.extend([self.channels[4], self.channels[4]])

        self.convs = nn.LayerList()
        self.upsamples = nn.LayerList()
        self.to_rgbs = nn.LayerList()
        self.noises = nn.Layer()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2**res, 2**res]
            self.noises.register_buffer(f"noise_{layer_idx}",
                                        paddle.randn(shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2**i]

            self.convs.append(
                StyledConv(
                    in_channel * 2 if is_concat else in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    is_concat=is_concat,
                ))

            self.convs.append(
                StyledConv(out_channel * 2 if is_concat else out_channel,
                           out_channel,
                           3,
                           style_dim,
                           blur_kernel=blur_kernel,
                           is_concat=is_concat))

            self.to_rgbs.append(
                ToRGB(out_channel * 2 if is_concat else out_channel, style_dim))

            self.channels_lst.extend([in_channel, out_channel, out_channel])
            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2
        self.is_concat = is_concat

    def make_noise(self):
        noises = [paddle.randn((1, 1, 2**2, 2**2))]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(paddle.randn((1, 1, 2**i, 2**i)))

        return noises

    def mean_latent(self, n_latent):
        latent_in = paddle.randn((n_latent, self.style_dim))
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, inputs):
        return self.style(inputs)

    def get_latents(
        self,
        inputs,
        truncation=1.0,
        truncation_cutoff=None,
        truncation_latent=None,
        input_is_latent=False,
    ):
        assert truncation >= 0, "truncation should be a float in range [0, 1]"

        if not input_is_latent:
            style = self.style(inputs)
        if truncation < 1.0:
            if truncation_latent is None:
                truncation_latent = self.get_mean_style()
            cutoff = truncation_cutoff
            if truncation_cutoff is None:
                style = truncation_latent + \
                    truncation * (style - truncation_latent)
            else:
                style[:, :cutoff] = truncation_latent[:, :cutoff] + \
                    truncation * (style[:, :cutoff] - truncation_latent[:, :cutoff])
        return style

    # @paddle.no_grad()
    def get_mean_style(self, n_sample=10, n_latent=1024):
        mean_style = None
        for i in range(n_sample):
            style = self.mean_latent(n_latent)
            if mean_style is None:
                mean_style = style
            else:
                mean_style += style

        mean_style /= n_sample
        return mean_style

    def get_latent_S(self, inputs):
        return self.style_affine(self.style(inputs))

    def style_affine(self, latent):
        if latent.ndim < 3:
            latent = latent.unsqueeze(1).tile((1, self.n_latent, 1))
        latent_ = []
        latent_.append(self.conv1.conv.modulation(latent[:, 0]))
        latent_.append(self.to_rgb1.conv.modulation(latent[:, 1]))

        i = 1
        for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2],
                                        self.to_rgbs):
            latent_.append(conv1.conv.modulation(latent[:, i + 0]))
            latent_.append(conv2.conv.modulation(latent[:, i + 1]))
            latent_.append(to_rgb.conv.modulation(latent[:, i + 2]))
            i += 2
        return latent_  #paddle.concat(latent_, axis=1)

    def synthesis(self,
                  latent,
                  noise=None,
                  randomize_noise=True,
                  is_w_latent=False):
        out = self.input(latent[0].shape[0])
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
                #noise = [paddle.randn(getattr(self.noises, f"noise_{i}").shape) for i in range(self.num_layers)]
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}")
                    for i in range(self.num_layers)
                ]

        out = self.conv1(out, latent[0], noise=noise[0])

        skip = self.to_rgb1(out, latent[1])

        i = 2
        if self.is_concat:
            noise_i = 1

            for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2],
                                            self.to_rgbs):
                out = conv1(out, latent[i],
                            noise=noise[(noise_i + 1) // 2])  ### 1 for 2
                out = conv2(out, latent[i + 1],
                            noise=noise[(noise_i + 2) // 2])  ### 1 for 2
                skip = to_rgb(out, latent[i + 2], skip)

                i += 3
                noise_i += 2
        else:
            for conv1, conv2, noise1, noise2, to_rgb in zip(
                    self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2],
                    self.to_rgbs):
                out = conv1(out, latent[i], noise=noise1)
                out = conv2(out, latent[i + 1], noise=noise2)
                skip = to_rgb(out, latent[i + 2], skip)

                i += 3

        return skip  #image = skip

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1.0,
        truncation_cutoff=None,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if truncation < 1.0:
            style_t = []
            if truncation_latent is None:
                truncation_latent = self.get_mean_style()
            cutoff = truncation_cutoff
            for style in styles:
                if truncation_cutoff is None:
                    style = truncation_latent + \
                        truncation * (style - truncation_latent)
                else:
                    style[:, :cutoff] = truncation_latent[:, :cutoff] + \
                    truncation * (style[:, :cutoff] - truncation_latent[:, :cutoff])
                style_t.append(style)
            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).tile((1, inject_index, 1))

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).tile((1, inject_index, 1))
            latent2 = styles[1].unsqueeze(1).tile(
                (1, self.n_latent - inject_index, 1))

            latent = paddle.concat([latent, latent2], 1)

        #if not input_is_affined_latent:
        styles = self.style_affine(latent)

        image = self.synthesis(styles, noise, randomize_noise)

        if return_latents:
            return image, latent
        else:
            return image, None
