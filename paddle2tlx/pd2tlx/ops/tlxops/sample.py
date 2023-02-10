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
import tensorlayerx as tlx
from paddle.nn import functional as F
import tensorlayerx.nn as nn

__all__ = [
    'tlx_Upsample',      # F.interpolate
    'tlx_PixelShuffle',  # F.pixel_shuffle
    'tlx_UpsamplingBilinear2d',
    "tlx_Identity", 
]


class tlx_Upsample(nn.Module):
    """python/paddle/nn/layer/common.py"""
    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=False,
                 align_mode=0,
                 data_format='channels_first',
                 name=None):
        super(tlx_Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode.lower()
        self.align_corners = align_corners
        self.align_mode = align_mode
        self.data_format = data_format
        self.name = name

    def forward(self, x):
        if self.data_format == "channels_first":
            data_format = "NCHW"
        else:
            data_format = "NHWC"
        out = tlx.ops.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
            data_format=data_format,
            name=self.name)

        return out

    def extra_repr(self):
        if self.scale_factor is not None:
            main_str = 'scale_factor={}'.format(self.scale_factor)
        else:
            main_str = 'size={}'.format(self.size)
        name_str = ', name={}'.format(self.name) if self.name else ''
        return '{}, mode={}, align_corners={}, align_mode={}, data_format={}{}'.format(
            main_str, self.mode, self.align_corners, self.align_mode,
            self.data_format, name_str)


class tlx_PixelShuffle(nn.Module):
    def __init__(self, upscale_factor, data_format="channels_last", name=None):
        super(tlx_PixelShuffle, self).__init__()

        if not isinstance(upscale_factor, int):
            raise TypeError("upscale factor must be int type")

        if data_format not in ["NCHW", "NHWC"]:
            raise ValueError("Data format should be 'NCHW' or 'NHWC'."
                             "But recevie data format: {}".format(data_format))
        self._upscale_factor = upscale_factor
        self._data_format = data_format
        self._name = name

    def extra_repr(self):
        main_str = 'upscale_factor={}'.format(self._upscale_factor)
        if self._data_format != 'channels_first':
            main_str += ', data_format={}'.format(self._data_format)
        if self._name is not None:
            main_str += ', name={}'.format(self._name)
        return main_str

    def forward(self, x):
        data_format = tlx.ops.channels_switching(self._data_format)
        return F.pixel_shuffle(x, self._upscale_factor,
                                        data_format, self._name)

class tlx_UpsamplingBilinear2d(nn.Module):
    """
    https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/layer/common.py
    """

    def __init__(
        self, size=None, scale_factor=None, data_format='channels_first', name=None
    ):
        super(tlx_UpsamplingBilinear2d, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.data_format = data_format
        self.name = name

    def forward(self, x):
        if self.data_format == "channels_first":
            data_format = 'NCHW'
        elif self.data_format == "channels_last":
            data_format = 'NHWC'        
        out = tlx.ops.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=True,
            align_mode=0,
            data_format=data_format,
            name=self.name,
        )

        return out

    def extra_repr(self):
        if self.scale_factor is not None:
            main_str = 'scale_factor={}'.format(self.scale_factor)
        else:
            main_str = 'size={}'.format(self.size)
        name_str = ', name={}'.format(self.name) if self.name else ''
        return '{}, data_format={}{}'.format(
            main_str, self.data_format, name_str
        )

class tlx_Identity(nn.Module):
    """
    A placeholder identity operator that accepts exactly one argument.
    """

    def __init__(self):
        super(tlx_Identity, self).__init__()

    def forward(self, x):
        return x