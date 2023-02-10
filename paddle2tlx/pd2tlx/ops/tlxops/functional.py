from __future__ import absolute_import, division, print_function
import numpy as np
import math
import numbers
import importlib
import cv2
# from tensorlayerx.backend.ops import convert_to_tensor
import collections
import sys
import paddle
from PIL import Image

from . import functional_pil as F_pil
from . import functional_cv2 as F_cv2
from . import functional_tensor as F_t
import tensorlayerx as tlx

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = [
        'tlx_hflip', 
        'tlx_vflip',
        'tlx_resize',
        'tlx_pad',
        'tlx_crop',
        'tlx_normalize', 
        'tlx_to_grayscale', 
        'tlx_to_tensor', 

        ]

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return isinstance(img, paddle.Tensor)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _get_image_size(img):
    if _is_pil_image(img):
    # if isinstance(img, Image.Image):
        return img.size
    elif _is_numpy_image(img):
    # elif isinstance(img, np.ndarray) and (img.ndim in {2, 3}):
        return img.shape[:2][::-1]
    elif _is_tensor_image(img):
    # elif isinstance(img, paddle.Tensor):
        return img.shape[1:][::-1]  # chw
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

def tlx_hflip(img):
    """Horizontally flips the given image.

    Args:
        image (np.array): Image to be flipped.

    Returns:
        np.array:  Horizontall flipped image.

    """

    if not (_is_pil_image(img) or _is_numpy_image(img) or
            _is_tensor_image(img)):
        raise TypeError(
            'img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.hflip(img)
    elif _is_tensor_image(img):
        return F_t.hflip(img)
    else:
        return F_cv2.hflip(img)


def tlx_vflip(img):
    """Vertically flips the given np.array.

    Args:
        image (np.array): Image to be flipped.

    Returns:
        np.array:  Vertically flipped image.

    """

    if not (_is_pil_image(img) or _is_numpy_image(img) or
            _is_tensor_image(img)):
        raise TypeError(
            'img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.vflip(img)
    elif _is_tensor_image(img):
        return F_t.vflip(img)
    else:
        return F_cv2.vflip(img)

def tlx_resize(img, size, interpolation='bilinear', data_format="CHW"):
    """
    Resizes the image to given size
    Args:
        input (PIL.Image|np.ndarray): Image to be resized.
        size (int|list|tuple): Target size of input data, with (height, width) shape.
        interpolation (int|str, optional): Interpolation method. when use pil backend, 
            support method are as following: 
            - "nearest": Image.NEAREST, 
            - "bilinear": Image.BILINEAR, 
            - "bicubic": Image.BICUBIC, 
            - "box": Image.BOX, 
            - "lanczos": Image.LANCZOS, 
            - "hamming": Image.HAMMING
            when use cv2 backend, support method are as following: 
            - "nearest": cv2.INTER_NEAREST, 
            - "bilinear": cv2.INTER_LINEAR, 
            - "area": cv2.INTER_AREA, 
            - "bicubic": cv2.INTER_CUBIC, 
            - "lanczos": cv2.INTER_LANCZOS4
    Returns:
        PIL.Image or np.array: Resized image.
    Examples:
        .. code-block:: python
            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F
            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')
            fake_img = Image.fromarray(fake_img)
            converted_img = F.resize(fake_img, 224)
            print(converted_img.size)
            # (262, 224)
            converted_img = F.resize(fake_img, (200, 150))
            print(converted_img.size)
            # (150, 200)
    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or
            _is_tensor_image(img)):
        raise TypeError(
            'img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.resize(img, size, interpolation)
    elif _is_tensor_image(img):
        return F_t.resize(img, size, interpolation)
    else:
        return F_cv2.resize(img, size, interpolation)

def tlx_pad(img, padding, fill=0, padding_mode='constant'):
    if not (_is_pil_image(img) or _is_numpy_image(img) or
            _is_tensor_image(img)):
        raise TypeError(
            'img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.pad(img, padding, fill, padding_mode)
    elif _is_tensor_image(img):
        return F_t.pad(img, padding, fill, padding_mode)
    else:
        return F_cv2.pad(img, padding, fill, padding_mode)

def tlx_crop(img, top, left, height, width):
    """Crops the given Image.
    Args:
        img (PIL.Image|np.array): Image to be cropped. (0,0) denotes the top left 
            corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
    Returns:
        PIL.Image or np.array: Cropped image.
    Examples:
        .. code-block:: python
            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F
            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')
            fake_img = Image.fromarray(fake_img)
            cropped_img = F.crop(fake_img, 56, 150, 200, 100)
            print(cropped_img.size)
    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or
            _is_tensor_image(img)):
        raise TypeError(
            'img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.crop(img, top, left, height, width)
    elif _is_tensor_image(img):
        return F_t.crop(img, top, left, height, width)
    else:
        return F_cv2.crop(img, top, left, height, width)

def tlx_normalize(img, mean, std, data_format='CHW', to_rgb=False):
    """Normalizes a tensor or image with mean and standard deviation.
    Args:
        img (PIL.Image|np.array|paddle.Tensor): input data to be normalized.
        mean (list|tuple): Sequence of means for each channel.
        std (list|tuple): Sequence of standard deviations for each channel.
        data_format (str, optional): Data format of input img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
        to_rgb (bool, optional): Whether to convert to rgb. If input is tensor, 
            this option will be igored. Default: False.
    Returns:
        np.ndarray or Tensor: Normalized mage. Data format is same as input img.
    
    Examples:
        .. code-block:: python
            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F
            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')
            fake_img = Image.fromarray(fake_img)
            mean = [127.5, 127.5, 127.5]
            std = [127.5, 127.5, 127.5]
            normalized_img = F.normalize(fake_img, mean, std, data_format='HWC')
            print(normalized_img.max(), normalized_img.min())
    """

    if _is_tensor_image(img):
        return F_t.normalize(img, mean, std, data_format)
    else:
        if _is_pil_image(img):
            img = np.array(img).astype(np.float32)

        return F_cv2.normalize(img, mean, std, data_format, to_rgb)


def tlx_to_grayscale(img, num_output_channels=1):
    """Converts image to grayscale version of image.
    Args:
        img (PIL.Image|np.array): Image to be converted to grayscale.
    Returns:
        PIL.Image or np.array: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel
            if num_output_channels = 3 : returned image is 3 channel with r = g = b
    
    Examples:
        .. code-block:: python
            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F
            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')
            fake_img = Image.fromarray(fake_img)
            gray_img = F.to_grayscale(fake_img)
            print(gray_img.size)
    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or
            _is_tensor_image(img)):
        raise TypeError(
            'img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.to_grayscale(img, num_output_channels)
    elif _is_tensor_image(img):
        return F_t.to_grayscale(img, num_output_channels)
    else:
        return F_cv2.to_grayscale(img, num_output_channels)



def tlx_to_tensor(pic, data_format='CHW'):
    """Converts a ``PIL.Image`` or ``numpy.ndarray`` to paddle.Tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (PIL.Image|np.ndarray): Image to be converted to tensor.
        data_format (str, optional): Data format of output tensor, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
    Returns:
        Tensor: Converted image. Data type is same as input img.
    Examples:
        .. code-block:: python
            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F
            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')
            fake_img = Image.fromarray(fake_img)
            tensor = F.to_tensor(fake_img)
            print(tensor.shape)
    """
    if not (_is_pil_image(pic) or _is_numpy_image(pic)
            or _is_tensor_image(pic)):
        raise TypeError(
            'pic should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'
            .format(type(pic)))

    if _is_pil_image(pic):
        return F_pil.to_tensor(pic, data_format)
    elif _is_numpy_image(pic):
        return F_cv2.to_tensor(pic, data_format)
    else:
        return pic if data_format.lower() == 'chw' else pic.transpose((1, 2, 0))

