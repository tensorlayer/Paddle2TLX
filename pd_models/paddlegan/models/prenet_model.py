#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

from .sr_model import BaseSRModel
from utils.visual import tensor2img
from .generators.builder import build_generator
from paddle2tlx.pd2tlx.utils import load_model_gan


MODEL_URLS = {
    "cyclegan": "../../../paddlegan/pretrain/PReNet.pdparams",
}


class PRe_Net(BaseSRModel):
    """PReNet Model.

    Paper: Progressive Image Deraining Networks: A Better and Simpler Baseline, IEEE,2019
    """

    def __init__(self, generator={"name":"PReNet"}):
        """Initialize the BasicVSR class.

        Args:
            generator (dict): config of generator.
            fix_iter (dict): config of fix_iter.
            pixel_criterion (dict): config of pixel criterion.
        """
        super(PRe_Net, self).__init__(generator)
        # self.nets['generator'] = build_generator(generator)

    def forward(self, input):
        model = self.nets['generator']
        model.eval()
        # with paddle.no_grad():
        output = self.nets['generator'](input)
        output = output[0, :, :, :].cpu()
        img = tensor2img(output)
        return img


def _prenet(pretrained=None):
    model = PRe_Net()
    if pretrained:
        model = load_model_gan(model, pretrained, "prenet")
    return model
