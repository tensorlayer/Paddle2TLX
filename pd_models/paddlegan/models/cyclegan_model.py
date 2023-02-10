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
import sys
import paddle
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from tensorlayerx.files.utils import assign_weights
from .base_model import BaseModel
from paddle2tlx.pd2tlx.utils import load_model_gan


MODEL_URLS = {
    "cyclegan": "https://paddlegan.bj.bcebos.com/models/CycleGAN_horse2zebra.pdparams",
}


class CycleGAN(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    def __init__(self,
                 generator={'name': 'ResnetGenerator', 'output_nc': 3, 'n_blocks': 9, 'ngf': 64, 'use_dropout': False, 'norm_type': 'instance', 'input_nc': 3},
                 discriminator= {'name': 'NLayerDiscriminator', 'ndf': 64, 'n_layers': 3, 'norm_type': 'instance', 'input_nc': 3},
                 direction='a2b'):
        """Initialize the CycleGAN class.

        Args:
            generator (dict): config of generator.
            discriminator (dict): config of discriminator.
            cycle_criterion (dict): config of cycle criterion.
        """
        super(CycleGAN, self).__init__()
        self.direction = direction
        # define generators
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.nets['netG_A'] = build_generator(generator)
        self.nets['netG_B'] = build_generator(generator)

        # define discriminators
        if discriminator:
            self.nets['netD_A'] = build_discriminator(discriminator)
            self.nets['netD_B'] = build_discriminator(discriminator)


    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.direction == 'a2b'

        if AtoB:
            if 'A' in input:
                # self.real_A = paddle.to_tensor(input['A'])
                self.real_A =  input['A']
            if 'B' in input:
                # self.real_B = paddle.to_tensor(input['B'])
                self.real_B =  input['B']
        else:
            if 'B' in input:
                # self.real_A = paddle.to_tensor(input['B'])
                self.real_A =  input['B']
            if 'A' in input:
                # self.real_B = paddle.to_tensor(input['A'])
                self.real_B = input['A']

        if 'A_paths' in input:
            self.image_paths = input['A_paths']
        elif 'B_paths' in input:
            self.image_paths = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if hasattr(self, 'real_A'):
            self.fake_B = self.nets['netG_A'](self.real_A)  # G_A(A)
            self.rec_A = self.nets['netG_B'](self.fake_B)  # G_B(G_A(A))

            # visual
            self.visual_items['real_A'] = self.real_A
            self.visual_items['fake_B'] = self.fake_B
            self.visual_items['rec_A'] = self.rec_A

        if hasattr(self, 'real_B'):
            self.fake_A = self.nets['netG_B'](self.real_B)  # G_B(B)
            self.rec_B = self.nets['netG_A'](self.fake_A)  # G_A(G_B(B))

            # visual
            self.visual_items['real_B'] = self.real_B
            self.visual_items['fake_A'] = self.fake_A
            self.visual_items['rec_B'] = self.rec_B
        return self.visual_items


def _cyclegan(pretrained=None):
    model = CycleGAN()
    if pretrained:
        model = load_model_gan(model, MODEL_URLS, "cyclegan")
    return model
