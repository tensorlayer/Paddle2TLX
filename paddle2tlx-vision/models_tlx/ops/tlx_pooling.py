import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module
import paddle.nn.functional as F
from .utils import channels_switching


class pd_AvgPool2D(Module):
    r"""
    This operation applies 2D average pooling over input features based on the input,
    and kernel_size, stride, padding parameters. Input(X) and Output(Out) are
    in NCHW format, where N is batch size, C is the number of channels,
    H is the height of the feature, and W is the width of the feature.
    Example:
        Input:
            X shape: :math:`(N, C, :math:`H_{in}`, :math:`W_{in}`)`
        Attr:
            kernel_size: ksize
        Output:
            Out shape: :math:`(N, C, :math:`H_{out}`, :math:`W_{out}`)`
        ..  math::
            Output(N_i, C_j, h, w)  = \frac{\sum_{m=0}^{ksize[0]-1} \sum_{n=0}^{ksize[1]-1}
                Input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)}{ksize[0] * ksize[1]}
    Parameters:
        kernel_size(int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be a square of an int.
        stride(int|list|tuple, optional): The pool stride size. If pool stride size is a tuple or list,
            it must contain two integers, (pool_stride_Height, pool_stride_Width).
            Otherwise, the pool stride size will be a square of an int.
            Default None, then stride will be equal to the kernel_size.
        padding(str|int|list|tuple, optional): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 2, [pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is 4. [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode(bool, optional): When True, will use `ceil` instead of `floor` to compute the output shape.
        exclusive(bool, optional): Whether to exclude padding points in average pooling
            mode, default is `true`.
        divisor_override(float, optional): If specified, it will be used as divisor, otherwise kernel_size will be
            used. Default None.
        data_format(str, optional): The data format of the input and output data. An optional string from: `"NCHW"`,
            `"NDHW"`. The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer to :ref:`api_guide_Name`.
            Usually name is no need to set and None by default.
    Shape:
        - x(Tensor): The input tensor of avg pool2d operator, which is a 4-D tensor.
          The data type can be float32, float64.
        - output(Tensor): The output tensor of avg pool2d  operator, which is a 4-D tensor.
          The data type is same as input x.
    Returns:
        A callable object of AvgPool2D.
    Examples:
        .. code-block:: python
            import paddle
            import paddle.nn as nn
            import numpy as np
            # max pool2d
            input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32]).astype(np.float32))
            AvgPool2D = nn.AvgPool2D(kernel_size=2,
                                stride=2, padding=0)
            output = AvgPool2D(input)
            # output.shape [1, 3, 16, 16]
    """

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        exclusive=True,
        divisor_override=None,
        data_format="NCHW",
        name=None,
    ):
        super(pd_AvgPool2D, self).__init__()
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.exclusive = exclusive
        self.divisor = divisor_override
        self.data_format = data_format
        self.name = name

    def forward(self, x):
        return F.avg_pool2d(
            x,
            kernel_size=self.ksize,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            exclusive=self.exclusive,
            divisor_override=self.divisor,
            data_format=self.data_format,
            name=self.name,
        )

    def extra_repr(self):
        return 'kernel_size={ksize}, stride={stride}, padding={padding}'.format(
            **self.__dict__
        )


class pd_MaxPool2D(Module):
    r"""
    This operation applies 2D max pooling over input feature based on the input,
    and kernel_size, stride, padding parameters. Input(X) and Output(Out) are
    in NCHW format, where N is batch size, C is the number of channels,
    H is the height of the feature, and W is the width of the feature.
    Example:
        - Input:
            X shape: :math:`(N, C, H_{in}, W_{in})`
        - Attr:
            kernel_size: ksize
        - Output:
            Out shape: :math:`(N, C, H_{out}, W_{out})`
        ..  math::
            Output(N_i, C_j, h, w) = \max_{m=0, \ldots, ksize[0] -1} \max_{n=0, \ldots, ksize[1]-1}
                Input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)
    Parameters:
        kernel_size(int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be a square of an int.
        stride(int|list|tuple, optional): The pool stride size. If pool stride size is a tuple or list,
            it must contain two integers, (pool_stride_Height, pool_stride_Width).
            Otherwise, the pool stride size will be a square of an int.
            Default None, then stride will be equal to the kernel_size.
        padding(str|int|list|tuple, optional): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 2, [pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is \4. [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode(bool, optional): when True, will use `ceil` instead of `floor` to compute the output shape
        return_mask(bool, optional): Whether to return the max indices along with the outputs.
        data_format(str, optional): The data format of the input and output data. An optional string from: `"NCHW"`, `"NDHW"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer to :ref:`api_guide_Name`.
            Usually name is no need to set and None by default.
    Returns:
        A callable object of MaxPool2D.
    Shape:
        - x(Tensor): The input tensor of max pool2d operator, which is a 4-D tensor.
          The data type can be float32, float64.
        - output(Tensor): The output tensor of max pool2d  operator, which is a 4-D tensor.
          The data type is same as input x.
    Examples:
        .. code-block:: python
            import paddle
            import paddle.nn as nn
            import numpy as np
            # max pool2d
            input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32]).astype(np.float32))
            MaxPool2D = nn.MaxPool2D(kernel_size=2,
                                   stride=2, padding=0)
            output = MaxPool2D(input)
            # output.shape [1, 3, 16, 16]
            # for return_mask=True
            MaxPool2D = nn.MaxPool2D(kernel_size=2, stride=2, padding=0, return_mask=True)
            output, max_indices = MaxPool2D(input)
            # output.shape [1, 3, 16, 16], max_indices.shape [1, 3, 16, 16],
    """

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        return_mask=False,
        ceil_mode=False,
        data_format="NCHW",
        name=None,
    ):
        super(pd_MaxPool2D, self).__init__()
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding
        self.return_mask = return_mask
        self.ceil_mode = ceil_mode
        self.data_format = data_format
        self.name = name

    def forward(self, x):
        return F.max_pool2d(
            x,
            kernel_size=self.ksize,
            stride=self.stride,
            padding=self.padding,
            return_mask=self.return_mask,
            ceil_mode=self.ceil_mode,
            data_format=self.data_format,
            name=self.name,
        )

    def extra_repr(self):
        return 'kernel_size={ksize}, stride={stride}, padding={padding}'.format(
            **self.__dict__
        )
