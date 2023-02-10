from __future__ import division

import math
import numbers

import paddle
import paddle.nn.functional as F
import tensorlayerx as tlx
import sys
import collections

__all__ = ["resize","pad","hflip","vflip","crop","normalize","to_grayscale"]

def _assert_image_tensor(img, data_format):
    if not isinstance(
            img, paddle.Tensor
    ) or img.ndim < 3 or img.ndim > 4 or not data_format.lower() in ('chw',
                                                                     'hwc'):
        raise RuntimeError(
            'not support [type={}, ndim={}, data_format={}] paddle image'.
            format(type(img), img.ndim, data_format))


def _get_image_h_axis(data_format):
    if data_format.lower() == 'chw':
        return -2
    elif data_format.lower() == 'hwc':
        return -3


def _get_image_w_axis(data_format):
    if data_format.lower() == 'chw':
        return -1
    elif data_format.lower() == 'hwc':
        return -2

def _get_image_c_axis(data_format):
    if data_format.lower() == 'chw':
        return -3
    elif data_format.lower() == 'hwc':
        return -1

def _get_image_size(img, data_format):
    return img.shape[_get_image_w_axis(data_format)], img.shape[
        _get_image_h_axis(data_format)]
    

def _is_channel_first(data_format):
    return _get_image_c_axis(data_format) == -3

def resize(img, size, interpolation='bilinear', data_format='CHW'):
    """
    Resizes the image to given size
    Args:
        input (paddle.Tensor): Image to be resized.
        size (int|list|tuple): Target size of input data, with (height, width) shape.
        interpolation (int|str, optional): Interpolation method. when use paddle backend, 
            support method are as following: 
            - "nearest"  
            - "bilinear"
            - "bicubic"
            - "trilinear"
            - "area"
            - "linear"
        data_format (str, optional): paddle.Tensor format
            - 'CHW'
            - 'HWC'
    Returns:
        paddle.Tensor: Resized image.
    """
    _assert_image_tensor(img, data_format)

    if not (isinstance(size, int) or
            (isinstance(size, (tuple, list)) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = _get_image_size(img, data_format)
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
    else:
        oh, ow = size

    img = img.unsqueeze(0)
    # img = F.interpolate(img,
    #                     size=(oh, ow),
    #                     mode=interpolation.lower(),
    #                     data_format='N' + data_format.upper())
    img = tlx.ops.interpolate(img,
                        size=(oh, ow),
                        mode=interpolation.lower(),
                        data_format='N' + data_format.upper())

    return img.squeeze(0)

def pad(img, padding, fill=0, padding_mode='constant', data_format='CHW'):
    """
    Pads the given paddle.Tensor on all sides with specified padding mode and fill value.
    Args:
        img (paddle.Tensor): Image to be padded.
        padding (int|list|tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (float, optional): Pixel fill value for constant fill. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant. Default: 0. 
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default: 'constant'.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value on the edge of the image
            - reflect: pads with reflection of image (without repeating the last value on the edge)
                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image (repeating the last value on the edge)
                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]
    Returns:
        paddle.Tensor: Padded image.
    """
    _assert_image_tensor(img, data_format)

    if not isinstance(padding, (numbers.Number, list, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, list, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, (list, tuple)) and len(padding) not in [2, 4]:
        raise ValueError(
            "Padding must be an int or a 2, or 4 element tuple, not a " +
            "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    else:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    padding = [pad_left, pad_right, pad_top, pad_bottom]

    if padding_mode == 'edge':
        padding_mode = 'replicate'
    elif padding_mode == 'symmetric':
        raise ValueError('Do not support symmetric mode')

    img = img.unsqueeze(0)
    #  'constant', 'reflect', 'replicate', 'circular'
    img = F.pad(img,
                pad=padding,
                mode=padding_mode,
                value=float(fill),
                data_format='N' + data_format)

    return img.squeeze(0)

def hflip(img, data_format='CHW'):
    """Horizontally flips the given paddle.Tensor Image.
    Args:
        img (paddle.Tensor): Image to be flipped.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
    Returns:
        paddle.Tensor:  Horizontall flipped image.
    """
    _assert_image_tensor(img, data_format)

    w_axis = _get_image_w_axis(data_format)

    return img.flip(axis=[w_axis])


def vflip(img, data_format='CHW'):
    """Vertically flips the given paddle tensor.
    Args:
        img (paddle.Tensor): Image to be flipped.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
    Returns:
        paddle.Tensor:  Vertically flipped image.
    """
    _assert_image_tensor(img, data_format)

    h_axis = _get_image_h_axis(data_format)

    return img.flip(axis=[h_axis])


def crop(img, top, left, height, width, data_format='CHW'):
    """Crops the given paddle.Tensor Image.
    Args:
        img (paddle.Tensor): Image to be cropped. (0,0) denotes the top left 
            corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
    Returns:
        paddle.Tensor: Cropped image.
    """
    _assert_image_tensor(img, data_format)

    if _is_channel_first(data_format):
        return img[:, top:top + height, left:left + width]
    else:
        return img[top:top + height, left:left + width, :]

def normalize(img, mean, std, data_format='CHW'):
    """Normalizes a tensor image given mean and standard deviation.
    Args:
        img (paddle.Tensor): input data to be normalized.
        mean (list|tuple): Sequence of means for each channel.
        std (list|tuple): Sequence of standard deviations for each channel.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
    Returns:
        Tensor: Normalized mage.
    """
    _assert_image_tensor(img, data_format)

    mean = paddle.to_tensor(mean, place=img.place)
    std = paddle.to_tensor(std, place=img.place)
    # mean = tlx.convert_to_tensor(mean)
    # std = tlx.convert_to_tensor(std)
    if _is_channel_first(data_format):
        mean = mean.reshape([-1, 1, 1])
        std = std.reshape([-1, 1, 1])

    return (img - mean) / std


def to_grayscale(img, num_output_channels=1, data_format='CHW'):
    """Converts image to grayscale version of image.
    Args:
        img (paddel.Tensor): Image to be converted to grayscale.
        num_output_channels (int, optionl[1, 3]):
            if num_output_channels = 1 : returned image is single channel
            if num_output_channels = 3 : returned image is 3 channel 
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
    Returns:
        paddle.Tensor: Grayscale version of the image.
    """
    _assert_image_tensor(img, data_format)

    if num_output_channels not in (1, 3):
        raise ValueError('num_output_channels should be either 1 or 3')

    rgb_weights = paddle.to_tensor([0.2989, 0.5870, 0.1140],
                                   place=img.place).astype(img.dtype)
    # rgb_weights = tlx.convert_to_tensor([0.2989, 0.5870, 0.1140]).astype(img.dtype)

    if _is_channel_first(data_format):
        rgb_weights = rgb_weights.reshape((-1, 1, 1))

    _c_index = _get_image_c_axis(data_format)

    img = (img * rgb_weights).sum(axis=_c_index, keepdim=True)
    _shape = img.shape
    _shape[_c_index] = num_output_channels

    return img.expand(_shape)