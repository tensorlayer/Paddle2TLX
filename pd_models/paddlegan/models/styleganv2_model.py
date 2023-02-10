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
import os
import math
import random
import numpy as np
import paddle
import paddle.nn as nn
from .base_model import BaseModel
from models.generators.builder import build_generator
from models.discriminators.builder import build_discriminator
from paddle2tlx.pd2tlx.utils import load_model_gan


MODEL_URLS = {
    "stylegan": "../../../paddlegan/pretrain/stylegan_v2_256_ffhq.pdparams",
}
generator_cfg={'name': 'StyleGANv2Generator', 'size': 256, 'style_dim': 512, 'n_mlp': 8}
discriminator_cfg={'name': 'StyleGANv2Discriminator', 'size': 256}

class StyleGAN2(BaseModel):
    """
    This class implements the StyleGANV2 model, for learning image-to-image translation without paired data.

    StyleGAN2 paper: https://arxiv.org/pdf/1912.04958.pdf
    """
    def __init__(self,
                 generator=generator_cfg,
                 discriminator=discriminator_cfg,
                 num_style_feat=512,
                 params=None):
        """Initialize the CycleGAN class.

        Args:
            generator (dict): config of generator.
            discriminator (dict): config of discriminator.
            gan_criterion (dict): config of gan criterion.
        """
        super(StyleGAN2, self).__init__(params)
        # self.mixing_prob = mixing_prob
        self.num_style_feat = num_style_feat
        # self.r1_reg_weight = r1_reg_weight

        # self.path_reg_weight = path_reg_weight
        # self.path_batch_shrink = path_batch_shrink
        # self.mean_path_length = 0

        self.nets['gen'] = build_generator(generator)

        # define discriminators
        if discriminator:
            self.nets['disc'] = build_discriminator(discriminator)

            self.nets['gen_ema'] = build_generator(generator)
            # self.model_ema(0)

            self.nets['gen'].train()
            self.nets['gen_ema'].eval()
            self.nets['disc'].train()
            # self.current_iter = 1

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with DataParallel.
        """
        # if isinstance(net, (paddle.DataParallel)):
        #     net = net._layers
        return net

    # def model_ema(self, decay=0.999):
    #     net_g = self.get_bare_model(self.nets['gen'])
    #     net_g_params = dict(net_g.named_parameters())

    #     neg_g_ema = self.get_bare_model(self.nets['gen_ema'])
    #     net_g_ema_params = dict(neg_g_ema.named_parameters())

    #     for k in net_g_ema_params.keys():
    #         net_g_ema_params[k].set_value(net_g_ema_params[k] * (decay) +
    #                                       (net_g_params[k] * (1 - decay)))

    def forward(self, x):
        self.nets['gen_ema'].eval()
        # batch = self.real_img.shape[0]
        batch = x.shape[0]
        noises = [paddle.randn([batch, self.num_style_feat])]
        fake_img, _ = self.nets['gen_ema'](noises)
        return fake_img


class InferGenerator(nn.Layer):
    def set_generator(self, generator):
        self.generator = generator

    def forward(self, style, truncation):
        truncation_latent = self.generator.get_mean_style()
        out = self.generator(styles=style,
                             truncation=truncation,
                             truncation_latent=truncation_latent)
        return out[0]


def _stylegan(pretrained=None):
    model = StyleGAN2()
    if pretrained:
        model = load_model_gan(model, pretrained, "stylegan")
    return model
