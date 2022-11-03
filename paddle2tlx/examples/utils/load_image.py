# coding: utf-8
import numpy as np
from PIL import Image


def load_image(image_path, mode=None):
    """
    data format: nchw
    Args:
        image_path:
        mode: None|tlx|pd|pt

    Returns:

    """
    # import torch
    from paddle import to_tensor
    import tensorlayerx as tlx

    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)

    if mode == "tlx":
        img = np.expand_dims(img, 0)
        # img = img.flatten()
        img = img / 255.0
        # img = to_tensor(img)
        img = tlx.convert_to_tensor(img)
        img = tlx.ops.nhwc_to_nchw(img)

        # img = tlx.vision.load_image(image_path)
        # img = tlx.vision.transforms.Resize((224, 224))(img).astype(np.float32) / 255
        # img = paddle.unsqueeze(paddle.Tensor(img), 0)
        # # print('img shape[nhwc]:', img.shape)
        # img = tlx.ops.nhwc_to_nchw(img)
        # # print('img shape[hchw]:', img.shape)

    elif mode == "pd":
        img = img.transpose((2, 0, 1))  # CHW
        # img = img[(2, 1, 0), :, :]  # BGR
        img = np.expand_dims(img, 0)
        # # img = img.flatten()
        img = img / 255.0
        img = to_tensor(img)

    elif mode == "pt":
        # img = img.transpose((2, 0, 1))
        # img = np.expand_dims(img, 0)
        # img = img / 255.0
        # img = torch.from_numpy(img)
        pass

    else:
        img = img.transpose((2, 0, 1))
        img = img / 255.0

    return img
