
from __future__ import division

import sys
import numbers
import collections

import numpy as np
import cv2
import tensorlayerx as tlx
from tensorlayerx.backend.ops.paddle_nn import convert_dtype
if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = ["to_tensor", "resize","pad","hflip","vflip","crop","normalize","to_grayscale"]

def to_tensor(pic, data_format='CHW'):
    """Converts a ``numpy.ndarray`` to paddle.Tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (np.ndarray): Image to be converted to tensor.
        data_format (str, optional): Data format of output tensor, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
    Returns:
        Tensor: Converted image.
    """

    if data_format not in ['CHW', 'HWC']:
        raise ValueError(
            'data_format should be CHW or HWC. Got {}'.format(data_format))

    if pic.ndim == 2:
        pic = pic[:, :, None]

    if data_format == 'CHW':
        img = tlx.convert_to_tensor(pic.transpose((2, 0, 1)))
    else:
        img = tlx.convert_to_tensor(pic)

    if convert_dtype(img.dtype) == 'uint8':
        return tlx.cast(img, np.float32) / 255.
    else:
        return 

def resize(img, size, interpolation='bilinear'):
    """
    Resizes the image to given size
    Args:
        input (np.ndarray): Image to be resized.
        size (int|list|tuple): Target size of input data, with (height, width) shape.
        interpolation (int|str, optional): Interpolation method. when use cv2 backend, 
            support method are as following: 
            - "nearest": cv2.INTER_NEAREST, 
            - "bilinear": cv2.INTER_LINEAR, 
            - "area": cv2.INTER_AREA, 
            - "bicubic": cv2.INTER_CUBIC, 
            - "lanczos": cv2.INTER_LANCZOS4
    Returns:
        np.array: Resized image.
    """
    _cv2_interp_from_str = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }

    if not (isinstance(size, int) or
            (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    h, w = img.shape[:2]

    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            output = cv2.resize(
                img,
                dsize=(ow, oh),
                interpolation=_cv2_interp_from_str[interpolation])
        else:
            oh = size
            ow = int(size * w / h)
            output = cv2.resize(
                img,
                dsize=(ow, oh),
                interpolation=_cv2_interp_from_str[interpolation])
    else:
        output = cv2.resize(img,
                            dsize=(size[1], size[0]),
                            interpolation=_cv2_interp_from_str[interpolation])
    if len(img.shape) == 3 and img.shape[2] == 1:
        return output[:, :, np.newaxis]
    else:
        return output

def pad(img, padding, fill=0, padding_mode='constant'):
    """
    Pads the given numpy.array on all sides with specified padding mode and fill value.
    Args:
        img (np.array): Image to be padded.
        padding (int|list|tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If list/tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a list/tuple of length 4 is provided
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
        np.array: Padded image.
    """
    _cv2_pad_from_str = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }

    if not isinstance(padding, (numbers.Number, list, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, list, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
        raise ValueError(
            "Padding must be an int or a 2, or 4 element tuple, not a " +
            "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, list):
        padding = tuple(padding)
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.copyMakeBorder(img,
                                  top=pad_top,
                                  bottom=pad_bottom,
                                  left=pad_left,
                                  right=pad_right,
                                  borderType=_cv2_pad_from_str[padding_mode],
                                  value=fill)[:, :, np.newaxis]
    else:
        return cv2.copyMakeBorder(img,
                                  top=pad_top,
                                  bottom=pad_bottom,
                                  left=pad_left,
                                  right=pad_right,
                                  borderType=_cv2_pad_from_str[padding_mode],
                                  value=fill)


def hflip(img):
    """Horizontally flips the given image.
    Args:
        img (np.array): Image to be flipped.
    Returns:
        np.array:  Horizontall flipped image.
    """
    return cv2.flip(img, 1)


def vflip(img):
    """Vertically flips the given np.array.
    Args:
        img (np.array): Image to be flipped.
    Returns:
        np.array:  Vertically flipped image.
    """
    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.flip(img, 0)[:, :, np.newaxis]
    else:
        return cv2.flip(img, 0)

def crop(img, top, left, height, width):
    """Crops the given image.
    Args:
        img (np.array): Image to be cropped. (0,0) denotes the top left 
            corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
    Returns:
        np.array: Cropped image.
    """

    return img[top:top + height, left:left + width, :]

def normalize(img, mean, std, data_format='CHW', to_rgb=False):
    """Normalizes a ndarray imge or image with mean and standard deviation.
    Args:
        img (np.array): input data to be normalized.
        mean (list|tuple): Sequence of means for each channel.
        std (list|tuple): Sequence of standard deviations for each channel.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
        to_rgb (bool, optional): Whether to convert to rgb. Default: False.
    Returns:
        np.array: Normalized mage.
    """

    if data_format == 'CHW':
        mean = np.float32(np.array(mean).reshape(-1, 1, 1))
        std = np.float32(np.array(std).reshape(-1, 1, 1))
    else:
        mean = np.float32(np.array(mean).reshape(1, 1, -1))
        std = np.float32(np.array(std).reshape(1, 1, -1))
    if to_rgb:
        # inplace
        img = img[..., ::-1]

    img = (img - mean) / std
    return img


def to_grayscale(img, num_output_channels=1):
    """Converts image to grayscale version of image.
    Args:
        img (np.array): Image to be converted to grayscale.
    Returns:
        np.array: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel
            if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if num_output_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    elif num_output_channels == 3:
        # much faster than doing cvtColor to go back to gray
        img = np.broadcast_to(
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis], img.shape)
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img
