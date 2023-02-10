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
import functools
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from .common import tlx_instance_norm

__all__ = [
    'tlx_InstanceNorm',      # F.instance_norm
    'tlx_InstanceNorm1d',    # F.instance_norm
    'tlx_InstanceNorm2d',    # F.instance_norm
    'tlx_InstanceNorm3d',    # F.instance_norm
]

class tlx_InstanceNorm(nn.Module):
    r"""
    Examples
    ---------
    With TensorLayerX

    >>> net = tlx.nn.Input([10, 50, 50, 32], name='input')
    >>> net = tlx.nn.BatchNorm()(net)

    Notes
    -----
    The :class:`BatchNorm` is universally suitable for 3D/4D/5D input in static model, but should not be used
    in dynamic model where layer is built upon class initialization. So the argument 'num_features' should only be used
    for subclasses :class:`BatchNorm1d`, :class:`BatchNorm2d` and :class:`BatchNorm3d`. All the three subclasses are
    suitable under all kinds of conditions.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`__
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`__

    """
    def __init__(self,
                 num_features,
                 epsilon=1e-5,
                #  momentum=0.9,
                 gamma_init = 'random_normal',
                 beta_init = 'zeros',
                 data_format="channels_first",
                 name=None,
                 act=None
                 ):
        super(tlx_InstanceNorm, self).__init__(name=name)
        # self.momentum = momentum
        self._epsilon = epsilon
        self.data_format = data_format
        self.num_features = num_features
        self.act = act
        self.gamma_init = self.str_to_init(gamma_init)
        self.beta_init = self.str_to_init(beta_init)
        self.axes = None
        # self.momentum = momentum
        # self.epsilon = epsilon
        # self.data_format = data_format
        # self.beta_init = self.str_to_init(beta_init)
        # self.gamma_init = self.str_to_init(gamma_init)
        # self.moving_mean_init = self.str_to_init(moving_mean_init)
        # self.moving_var_init = self.str_to_init(moving_var_init)
        # self.num_features = num_features
        # self.is_train = is_train

        self.axes = None
        if self.num_features:
            self.build(None)
            self._built = True

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(num_features={num_features}, momentum={momentum}' ', epsilon={epsilon}')
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name="{name}"'
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)


    def _get_param_shape(self, inputs_shape):
        if self.data_format == 'channels_last':
            axis = -1
        elif self.data_format == 'channels_first':
            axis = 1
        else:
            raise ValueError('data_format should be either %s or %s' % ('channels_last', 'channels_first'))

        channels = inputs_shape[axis]
        params_shape = [channels]

        return params_shape

    def _check_input_shape(self, inputs):
        if inputs.ndim <= 1:
            raise ValueError('expected input at least 2D, but got {}D input'.format(inputs.ndim))

    def build(self, inputs_shape):
        params_shape = [self.num_features] if self.num_features is not None else self._get_param_shape(inputs_shape)
        self.num_features = self.num_features if self.num_features is not None else params_shape[0]
        #  weight=self.scale, bias=self.bias, eps=self._epsilon
        self.biases, self.filters = None, None
        if self.beta_init:
            # self.beta = self._get_weights(var_name="beta", shape=params_shape, init=self.beta_init)
            self.biases = self._get_weights(var_name="biases", shape=params_shape, init=self.beta_init)
        if self.gamma_init:
            # self.gamma = self._get_weights(var_name="gamma", shape=params_shape, init=self.gamma_init)
            self.filters = self._get_weights(var_name="filters", shape=params_shape, init=self.gamma_init)

    def forward(self, inputs):
        self._check_input_shape(inputs)
        if self._forward_state == False:
            if self._built == False:
                self.build(tlx.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        # self.batchnorm = tlx.ops.InstanceNorm(
        #     decay=self.momentum, epsilon=self.epsilon, beta=self.beta, gamma=self.gamma, moving_mean=self.moving_mean,
        #     moving_var=self.moving_var, num_features=self.num_features, data_format=self.data_format, is_train=False
        # )
        # print(f"tlx_InstanceNorm.forward.data_format={self.data_format}")
        self.instancenorm = tlx_instance_norm(num_features=self.num_features, filters=self.filters, biases=self.biases, epsilon=self._epsilon, data_format=self.data_format)
        outputs = self.instancenorm(inputs=inputs)
        return outputs

class tlx_InstanceNorm1d(tlx_InstanceNorm):
    """The :class:`BatchNorm1d` applies Batch Normalization over 2D/3D input (a mini-batch of 1D
    inputs (optional) with additional channel dimension), of shape (N, C) or (N, L, C) or (N, C, L).
    See more details in :class:`BatchNorm`.

    Examples
    ---------
    With TensorLayerX

    >>> # in static model, no need to specify num_features
    >>> net = tlx.nn.Input([10, 50, 32], name='input')
    >>> net = tlx.nn.BatchNorm1d()(net)
    >>> # in dynamic model, build by specifying num_features
    >>> conv = tlx.nn.Conv1d(32, 5, 1, in_channels=3)
    >>> bn = tlx.nn.BatchNorm1d(num_features=32)

    """

    def _check_input_shape(self, inputs):
        if len(inputs.shape) != 2 and len(inputs.shape) != 3:
            raise ValueError('expected input to be 2D or 3D, but got {}D input'.format(inputs.ndim))


class tlx_InstanceNorm2d(tlx_InstanceNorm):
    """The :class:`BatchNorm2d` applies Batch Normalization over 4D input (a mini-batch of 2D
    inputs with additional channel dimension) of shape (N, H, W, C) or (N, C, H, W).
    See more details in :class:`BatchNorm`.

    Examples
    ---------
    With TensorLayer

    >>> # in static model, no need to specify num_features
    >>> net = tlx.nn.Input([10, 50, 50, 32], name='input')
    >>> net = tlx.nn.BatchNorm2d()(net)
    >>> # in dynamic model, build by specifying num_features
    >>> conv = tlx.nn.Conv2d(32, (5, 5), (1, 1), in_channels=3)
    >>> bn = tlx.nn.BatchNorm2d(num_features=32)

    """

    def _check_input_shape(self, inputs):
        if len(inputs.shape) != 4:
            raise ValueError('expected input to be 4D, but got {}D input'.format(inputs.ndim))


class tlx_InstanceNorm3d(tlx_InstanceNorm):
    """The :class:`BatchNorm3d` applies Batch Normalization over 5D input (a mini-batch of 3D
    inputs with additional channel dimension) with shape (N, D, H, W, C) or (N, C, D, H, W).
    See more details in :class:`BatchNorm`.

    Examples
    ---------
    With TensorLayer

    >>> # in static model, no need to specify num_features
    >>> net = tlx.nn.Input([10, 50, 50, 50, 32], name='input')
    >>> net = tlx.nn.BatchNorm3d()(net)
    >>> # in dynamic model, build by specifying num_features
    >>> conv = tlx.nn.Conv3d(32, (5, 5, 5), (1, 1), in_channels=3)
    >>> bn = tlx.nn.BatchNorm3d(num_features=32)

    """

    def _check_input_shape(self, inputs):
        if len(inputs.shape) != 5:
            raise ValueError('expected input to be 5D, but got {}D input'.format(inputs.ndim))


