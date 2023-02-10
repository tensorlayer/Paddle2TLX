import os
import sys
sys.path.append("")
from models import deeplab, farseg, unet
import numpy as np
from PIL import Image
import argparse
import paddle


img_path = "../../examples/images/LC80010812013365LGN00_18_photo.png"


def load_image(image_path):
    """
    data format: nchw
    Args:
        image_path:
        mode: None|tlx|pd|pt
    Returns:
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256), Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)
    img = img.transpose((2, 0, 1))  # CHW
    img = np.expand_dims(img, 0)
    img = img / 255.0
    img = paddle.to_tensor(img)
    return img


def parse_args():
    parser = argparse.ArgumentParser(description='PaddleRS-predict')
    parser.add_argument('--model_name',
                        type=str,
                        default="farseg",
                        help='train model name')
    # parser.add_argument('--img_path',
    #                     type=str,
    #                     default=None,
    #                     help='the test image')
    # parser.add_argument('--pretrain',
    #                     type=str,
    #                     default=None,
    #                     help='pretrain model path')
    args = parser.parse_args()
    return args


def predict(model_name):
    if model_name == "unet":
        model = unet._unet(True)
    elif model_name == "farseg":
        model = farseg._farseg(pretrained='../../pretrain/paddlersseg/farseg/model.pdparams')
    elif model_name == "deeplabv3p":
        model = deeplab._deeplabv3p(True)
    model.eval()
    img = load_image(img_path)
    results = model(img)[0]
    return results


if __name__ == "__main__":
    args = parse_args()
    if args.model_name not in ["farseg", "unet", "deeplabv3p"]:
        raise ValueError(f"no model_name={args.model_name}")
    res = predict(args.model_name)
    print(f"{args.model_name} paddle res={res}")
