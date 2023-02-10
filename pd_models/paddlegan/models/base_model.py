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

# code was heavily based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import numpy as np
from collections import OrderedDict
from abc import ABC, abstractmethod


class BaseModel(ABC):
    r"""This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:          initialize the class.
        -- <setup_input>:       unpack data from dataset and apply preprocessing.
        -- <forward>:           produce intermediate results.
        -- <train_iter>:        calculate losses, gradients, and update network weights.

    # trainer training logic:
    #
    #                build_model                               ||    model(BaseModel)
    #                     |                                    ||
    #               build_dataloader                           ||    dataloader
    #                     |                                    ||
    #               model.setup_lr_schedulers                  ||    lr_scheduler
    #                     |                                    ||
    #               model.setup_optimizers                     ||    optimizers
    #                     |                                    ||
    #     train loop (model.setup_input + model.train_iter)    ||    train loop
    #                     |                                    ||
    #         print log (model.get_current_losses)             ||
    #                     |                                    ||
    #         save checkpoint (model.nets)                     \/

    """
    def __init__(self, params=None):
        """Initialize the BaseModel class.

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <super(YourClass, self).__init__(self, cfg)>
        Then, you need to define four lists:
            -- self.losses (dict):          specify the training losses that you want to plot and save.
            -- self.nets (dict):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (dict):    define and initialize optimizers. You can define one optimizer for each network.
                                          If two networks are updated at the same time, you can use itertools.chain to group them.
                                          See cycle_gan_model.py for an example.

        Args:
            params (dict): Hyper params for train or test. Default: None.
        """
        self.params = params
        self.is_train = True if self.params is None else self.params.get(
            'is_train', True)

        self.nets = OrderedDict()
        self.visual_items = OrderedDict()

    # @abstractmethod
    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <train_iter> and <test_iter>."""
        pass