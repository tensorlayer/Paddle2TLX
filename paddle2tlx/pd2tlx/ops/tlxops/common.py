import tensorlayerx as tlx
import paddle.nn.functional as F
import random
from tensorlayerx.backend.ops.paddle_nn import framework


__all__ = [
    'convert_to_list',       # F.max_pool2d
    '_Pad2dbase',            # pd.vision.ops.deform_conv2d,...
    '_Conv2d_transposbase',  # F.conv2d_transpose
    'tlx_instance_norm',     # instance_norm
    'tlx_randperm',
    'tlx_linspace',
    'tlx_get_tensor_shape',
    'tlx_linear',
    'tlx_one_hot',
    'tlx_adaptive_max_pool2d',
    'tlx_adaptive_avg_pool2d'
]


def tlx_adaptive_max_pool2d(x, output_size, data_format="NCHW"):
    adaptivemaxpool2d = tlx.ops.AdaptiveMaxPool2D(output_size=output_size, data_format=data_format)
    outputs = adaptivemaxpool2d(inputs=x)
    return outputs


def tlx_adaptive_avg_pool2d(x, output_size, data_format="NCHW"):
    adaptivemeanpool2d = tlx.ops.AdaptiveMeanPool2D(output_size=output_size, data_format=data_format)
    outputs = adaptivemeanpool2d(inputs=x)
    return outputs


def convert_to_list(value, n, name, dtype=int):
    """
    Converts a single numerical type or iterable of numerical
    types into an numerical type list.
    Arguments:
      value: The value to validate and convert. Could an int, or any iterable
        of ints.
      n: The size of the list to be returned.
      name: The name of the argument being validated, e.g. "stride" or
        "filter_size". This is only used to format error messages.
      dtype: the numerical type of the element of the list to be returned.
    Returns:
      A list of n dtypes.
    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, dtype):
        return [
            value,
        ] * n
    else:
        try:
            value_list = list(value)
        except TypeError:
            raise ValueError("The " + name +
                             "'s type must be list or tuple. Received: " +
                             str(value))
        if len(value_list) != n:
            raise ValueError("The " + name + "'s length must be " + str(n) +
                             ". Received: " + str(value))
        for single_value in value_list:
            assert not isinstance(
                single_value, framework.Variable
            ), "Required numerical type with '%s', but received Tensor." % dtype
            try:
                dtype(single_value)
            except (ValueError, TypeError):
                raise ValueError("The " + name +
                                 "'s type must be a list or tuple of " +
                                 str(n) + " " + str(dtype) + " . Received: " +
                                 str(value) + " "
                                 "including element " + str(single_value) +
                                 " of type" + " " + str(type(single_value)))
        return value_list


def channels_switching(data_format, dim='2d', padding=None):
    if dim == '1d':
        if data_format == 'channels_first':
            out = 'NCL'
        if data_format == 'channels_last':
            out = 'NLC'
        pads = padding
    if dim == '2d':
        if data_format == 'channels_first':
            out = 'NCHW'
        if data_format == 'channels_last':
            out = 'NHWC'
        pads = [padding[1][0], padding[1][1], padding[0][0], padding[0][1]]
    if dim == '3d':
        if data_format == 'channels_first':
            out = 'NCDHW'
        if data_format == 'channels_last':
            out = 'NDHWC'
        pads = [padding[2][0], padding[2][1],
                padding[1][0], padding[1][1],
                padding[0][0], padding[0][1]]
    return out, pads


class _Pad2dbase(object):
    def __init__(self, padding, value=0.0, mode="constant", data_format="channels_last", name=None):
        # tensorlayerx/nn/layers/padding.py
        # self._pad = nn.layers.padding._npairs(padding, 2)
        self.pad = padding
        self.mode = mode
        self.value = value
        self.data_format = data_format
        self._name = name

    def __call__(self, inputs):
        if self.data_format =="channels_first":
            data_format="NCHW"
        if self.data_format == 'channels_last':
            data_format = 'NHWC'
        if isinstance(self.pad, int):
            padding = [self.pad, self.pad,self.pad, self.pad]
        elif isinstance(self.pad, tuple):
            padding = [self.pad[1][0], self.pad[1][1], self.pad[0][0], self.pad[0][1]]
        elif isinstance(self.pad, list):
            padding = self.pad
        # data_format, padding = tlx.ops.paddle_backend.channels_switching(self.data_format, '2d', self.pad)
        # data_format = channels_switching(self._data_format, '2d', self.padding)

        # out = F.pad(inputs, padding, mode=self.mode, value=self.value , data_format=data_format)
        return F.pad(inputs,
                     pad=padding,
                     mode=self.mode,
                     value=self.value,
                     data_format=data_format,
                     name=self._name)


# class tlx_ZeroPadding2D(object):
# # class tlx_ZeroPadding2D(object):
#     def __init__(self, padding, data_format, mode="constant"):
#         self.padding = padding
#         self.data_format = data_format
#         self.mode = mode

#     def __call__(self, inputs):
#         # data_format, padding = tlx.ops.channels_switching(self.data_format, '2d', self.padding)
#         data_format, padding = tlx.ops.channels_switching(self.data_format, '2d', self.padding)
#         # out = F.pad(inputs, padding, mode='constant', value=0.0, data_format=data_format)
#         # data_format="NCHW", padding=[[1,1],[2,3]]
#         # print(f"data_format={data_format}")
#         # print(f"padding={padding}")
#         out = F.pad(inputs, padding, mode=self.mode, value=0.0, data_format=data_format)
#         # print(f"tlx_ZeroPadding2D.out={len(out[0])}")
#         return out


class _Conv2d_transposbase(object):

    def __init__(
        self, strides, padding, output_padding=0, groups=1, data_format='channels_last', dilations=None, name=None, out_channel=None, k_size=None,
        in_channels=None
    ):
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = tlx.ops.preprocess_2d_format(data_format, padding)
        self.output_padding = output_padding
        self.groups = groups

    def __call__(self, input, filters,output_size):
        output = F.conv2d_transpose(
            x=input, weight=filters, stride=self.strides, padding=self.padding, output_padding=self.output_padding,dilation=self.dilations,
            data_format=self.data_format, groups=self.groups,
            output_size=output_size
        )
        return output


class tlx_instance_norm(object):
    def __init__(self,
        num_features,
        epsilon=1e-5,
        momentum=0.9,
        # gamma=None,
        # beta=None,
        filters= None,
        biases = None,
        data_format="channels_last"):

        self.epsilon = epsilon
        self.filters = filters
        self.biases = biases
        self.num_features = num_features
        self.data_format = data_format


    def __call__(self, inputs):
        data_format = self.channel_format(inputs)
        # print(f"_InstanceNormbase.__call__.data_format={data_format}")
        # outputs = pd.nn.functional.instance_norm(input, weight=self.scale, bias=self.bias, eps=self._epsilon)
        outputs = F.instance_norm(inputs, weight=self.filters, bias=self.biases, eps=self.epsilon, 
                                                            data_format=data_format)
        return outputs

    def channel_format(self, inputs):
        """ return "NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC" or "NDHWC". """
        len_in_shape = len(inputs.shape)
        if len_in_shape == 2:
            return 'NC'
        if self.data_format == 'channels_last':
            if len_in_shape == 3:
                return 'NLC'
            if len_in_shape == 4:
                return 'NHWC'
            if len_in_shape == 5:
                return 'NDHWC'
        if self.data_format == 'channels_first':
            if len_in_shape == 3:
                return 'NCL'
            if len_in_shape == 4:
                return 'NCHW'
            if len_in_shape == 5:
                return 'NCDHW'


def tlx_randperm(n, dtype="int64"):
    range_list = list(range(n))
    random.shuffle(range_list)
    return tlx.convert_to_tensor(range_list, dtype)


def tlx_linspace(start, stop, num, dtype=None):
    x = tlx.linspace(start, stop, num)
    if dtype is None:  # todo - det - picodet_lcnet
        dtype = x.dtype
    out = tlx.cast(x, dtype=dtype)
    return out

# def tlx_get_tensor_shape(tensor, dtype=None):
def tlx_get_tensor_shape(tensor):
    # dtype="int32" ==> detection
    x = tlx.get_tensor_shape(tensor)
    return tlx.convert_to_tensor(x, dtype='int32')


def tlx_linear(x, weight=None, bias=None, name=None, data_format='channels_first'):
    w_matmul = tlx.ops.MatMul()
    bias_add = tlx.ops.BiasAdd(data_format=data_format)
    if weight is not None:
        z = w_matmul(x, weight)
    if bias is not None:
        z = bias_add(z, bias)
    return z

def tlx_one_hot(x, num_classes, name=None):
    _onehot = tlx.ops.OneHot(depth=num_classes)(x)
    return _onehot


def tlx_nonzero(x, as_tuple=False):
    _nonzero = tlx.ops.CountNonzero()
    return _nonzero(x)