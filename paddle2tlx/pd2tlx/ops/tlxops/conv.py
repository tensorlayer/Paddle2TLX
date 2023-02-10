# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import paddle as pd
import tensorlayerx.nn as nn
from paddle.nn import functional as F
import numpy as np
import tensorlayerx as tlx
from tensorlayerx import logging
from .common  import _Conv2d_transposbase, convert_to_list

__all__ = [
    'tlx_DeformConv2d',    # pd.vision.ops.deform_conv2d,...
    'tlx_ConvTranspose2d', # F.conv2d_transpose
    # 'tlx_SpectralNorm'
]

class tlx_DeformConv2d(nn.Module):
    r"""
    python/paddle/vision/ops.py
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 deformable_groups=1,
                 groups=1,
                 W_init='random_normal',
                 b_init='zeros',
                 data_format="channels_first"):
        super(tlx_DeformConv2d, self).__init__()
        assert (
            W_init is not None
        ), "weight_attr should not be None in Conv."
        self._weight_attr = W_init
        self._bias_attr = b_init
        self._deformable_groups = deformable_groups
        self._groups = groups
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._channel_dim = 1
        self.kernel_size = kernel_size
        self._data_format = data_format
        self._stride = convert_to_list(stride, 2, 'stride')
        self._dilation = convert_to_list(dilation, 2, 'dilation')
        self._kernel_size = convert_to_list(kernel_size, 2, 'kernel_size')
        self._filter_init = self.str_to_init(W_init)
        self._bias_init = self.str_to_init(b_init)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")
        self._padding = convert_to_list(padding, 2, 'padding')
        self._filter_shape = [out_channels, in_channels // groups] + self._kernel_size
        self.build()

    def __repr__(self):
        pass


    def build(self):
        # TODO : check
        self.filters, self.biases = None, None
        # filter_shape = [self._out_channels, self._in_channels // self._groups] + self._kernel_size
        filter_shape = (
                     self.kernel_size, self.kernel_size , self._in_channels// self._groups , self._out_channels
                )
        # new_shape = [self._filter_shape[2],self._filter_shape[3],self._filter_shape[0],self._filter_shape[1]]
        # print(f"filter_shape={filter_shape}")
        if self._weight_attr is not None:
            self.filters = self._get_weights("filters", shape=filter_shape, init=self._filter_init)
            # print(f"self.filters.shape={self.filters.shape}")

        if self._bias_attr is not None:
            self.biases = self._get_weights("biases", shape=(self._out_channels, ), init=self._bias_init)


    def forward(self, inputs, offset, mask=None):
        return pd.vision.ops.deform_conv2d(
            x=inputs,
            offset=offset,
            weight=self.filters,
            bias=self.biases,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            deformable_groups=self._deformable_groups,
            groups=self._groups,
            mask=mask,
        )

class tlx_ConvTranspose2d(nn.Module):
    """Applies a 2D transposed convolution operator over an input image composed of several input planes.

    Parameters
    ----------
    out_channels : int
        Number of channels produced by the convolution
    kernel_size : tuple or int
        The kernel size (height, width).
    stride : tuple or int
        The sliding window stride of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    dilation : tuple or int
        Specifying the dilation rate to use for dilated convolution.
    act : activation function
        The activation function of this layer.
    padding : int, tuple or str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    W_init : initializer or str
        The initializer for the the kernel weight matrix.
    b_init : initializer or None or str
        The initializer for the the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayerx

    >>> net = tlx.nn.Input([8, 400, 400, 3], name='input')
    >>> conv2d_transpose = tlx.nn.ConvTranspose2d(out_channels=32, kernel_size=(3, 3), stride=(2, 2), b_init=None, in_channels=3, name='conv2d_transpose_1')
    >>> print(conv2d_transpose)
    >>> tensor = tlx.nn.ConvTranspose2d(out_channels=32, kernel_size=(3, 3), stride=(2, 2), act=tlx.ReLU, name='conv2d_transpose_2')(net)
    >>> print(tensor)

    """

    def __init__(
        self,
        out_channels=32,
        kernel_size=(3, 3),
        stride=(1, 1),
        act=None,
        padding='SAME',
        output_padding=0,
        data_format='channels_first',
        dilation=(1, 1),
        n_group=1,
        W_init='truncated_normal',
        b_init='constant',
        in_channels=None,
        name=None,  # 'conv2d_transpose',
    ):
        super(tlx_ConvTranspose2d, self).__init__(name, act=act)
        self.out_channels = out_channels
        self.kernel_size = self.check_param(kernel_size)
        self.stride = self.check_param(stride)
        self.padding = padding
        self.output_padding = output_padding
        self.groups = n_group
        self.data_format = data_format
        self.dilation = self.check_param(dilation)
        self.W_init = self.str_to_init(W_init)
        self.b_init = self.str_to_init(b_init)
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "ConvTranspose2d %s: out_channels: %d kernel_size: %s stride: %s pad: %s act: %s" % (
                self.name, out_channels, str(kernel_size), str(stride), padding,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}'
            ', stride={stride}, padding={padding}'
        )
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.data_format == 'channels_last':
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
        elif self.data_format == 'channels_first':
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        #TODO channels first filter shape [out_channel, in_channel, filter_h, filter_w]
        self.filter_shape = (self.kernel_size[0], self.kernel_size[1], self.out_channels//self.groups, self.in_channels)
        self.filters = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)#, transposed=True)

        self.b_init_flag = False
        if self.b_init:
            self.biases = self._get_weights("biases", shape=(self.out_channels, ), init=self.b_init)
            self.bias_add = tlx.ops.BiasAdd(self.data_format)
            self.b_init_flag = True

        # self.conv2d_transpose = tlx.ops.Conv2d_transpose(
        #     strides=self.stride, padding=self.padding, data_format=self.data_format, dilations=self.dilation,
        #     out_channel=self.out_channels, k_size=(self.kernel_size[0], self.kernel_size[1]), in_channels=self.in_channels
        # )
        self.conv2d_transpose = _Conv2d_transposbase(
            strides=self.stride, padding=self.padding, output_padding=self.output_padding, groups=self.groups,
            data_format=self.data_format, dilations=self.dilation,
            out_channel=self.out_channels, k_size=(self.kernel_size[0], self.kernel_size[1]),
            in_channels=self.in_channels
        )

        self.act_init_flag = False
        if self.act:
            self.act_init_flag = True

    def forward(self, inputs, output_size=None):
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        outputs = self.conv2d_transpose(inputs, self.filters,output_size)
        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.biases)
        if self.act_init_flag:
            outputs = self.act(outputs)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs
