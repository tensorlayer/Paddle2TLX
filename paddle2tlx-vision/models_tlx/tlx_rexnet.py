from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

os.environ['TL_BACKEND'] = 'paddle'
import numpy as np
import paddle
from paddle import ParamAttr
import tensorlayerx.nn as nn
from math import ceil
from utils.download import get_weights_path_from_url

from utils.load_model import restore_model

model_urls = {
    "rexnet":
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_0_pretrained.pdparams"
}

__all__ = []


def conv_bn_act(out,
                in_channels,
                channels,
                kernel=1,
                stride=1,
                pad=0,
                num_group=1,
                active=True,
                relu6=False):
    ######################################################################################################################################################################################################################################################################################################################
    out.append(
        nn.GroupConv2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            n_group=num_group,
            b_init=None,
            data_format='channels_first'
        ))
    # out.append(
    #     nn.Conv2D(
    #         in_channels,
    #         channels,
    #         kernel,
    #         stride,
    #         pad,
    #         groups=num_group,
    #         bias_attr=False))
    out.append(nn.BatchNorm2d(num_features=channels, data_format='channels_first'))
    if active:
        out.append(nn.ReLU6() if relu6 else nn.ReLU())


def conv_bn_swish(out,
                  in_channels,
                  channels,
                  kernel=1,
                  stride=1,
                  pad=0,
                  num_group=1):
    ######################################################################################################################################################################################################################################################################################################################
    out.append(
        nn.GroupConv2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=kernel,
            stride=stride,
            n_group=num_group,
            padding=pad,
            b_init=None,
            data_format='channels_first'
        ))
    # out.append(
    #     nn.Conv2D(
    #         in_channels,
    #         channels,
    #         kernel,
    #         stride,
    #         pad,
    #         groups=num_group,
    #         bias_attr=False))
    out.append(nn.BatchNorm2d(num_features=channels, data_format='channels_first'))
    out.append(nn.Swish())


class SE(nn.Module):
    def __init__(self, in_channels, channels, se_ratio=12):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1, data_format='channels_first')
        self.fc = nn.Sequential(
            ######################################################################################################################################################################################################################################################################################################################
            nn.Conv2d(
                in_channels=in_channels, out_channels=channels // se_ratio, kernel_size=1,
                data_format='channels_first'),
            # nn.Conv2D(
            #     in_channels, channels // se_ratio, kernel_size=1, padding=0),

            nn.BatchNorm2d(num_features=channels // se_ratio, data_format='channels_first'),
            nn.ReLU(),
            ##################################################################################################################################################################################################################################################################################################################################################################################################################################################
            nn.Conv2d(
                in_channels=channels // se_ratio, out_channels=channels, kernel_size=1,
                data_format='channels_first'),
            # nn.Conv2D(
            #     channels // se_ratio, channels, kernel_size=1, padding=0),

            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class LinearBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 t,
                 stride,
                 use_se=True,
                 se_ratio=12,
                 **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels

        out = []
        if t != 1:
            dw_channels = in_channels * t
            conv_bn_swish(out, in_channels=in_channels, channels=dw_channels)
        else:
            dw_channels = in_channels

        conv_bn_act(
            out,
            in_channels=dw_channels,
            channels=dw_channels,
            kernel=3,
            stride=stride,
            pad=1,
            num_group=dw_channels,
            active=False)

        if use_se:
            out.append(SE(dw_channels, dw_channels, se_ratio))

        out.append(nn.ReLU6())
        conv_bn_act(
            out,
            in_channels=dw_channels,
            channels=channels,
            active=False,
            relu6=True)
        self.out = nn.Sequential(*out)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out[:, 0:self.in_channels] += x
        return out


class ReXNetV1(nn.Module):
    def __init__(self,
                 input_ch=16,
                 final_ch=180,
                 width_mult=1.0,
                 depth_mult=1.0,
                 class_num=1000,
                 use_se=True,
                 se_ratio=12,
                 dropout_ratio=0.2,
                 bn_momentum=0.9):
        super(ReXNetV1, self).__init__()

        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        use_ses = [False, False, True, True, True, True]

        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1)
                       for idx, element in enumerate(strides)], [])
        if use_se:
            use_ses = sum([[element] * layers[idx]
                           for idx, element in enumerate(use_ses)], [])
        else:
            use_ses = [False] * sum(layers[:])
        ts = [1] * layers[0] + [6] * sum(layers[1:])

        self.depth = sum(layers[:]) * 3
        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = input_ch / width_mult if width_mult < 1.0 else input_ch

        features = []
        in_channels_group = []
        channels_group = []

        # The following channel configuration is a simple instance to make each layer become an expand layer.
        for i in range(self.depth // 3):
            if i == 0:
                in_channels_group.append(int(round(stem_channel * width_mult)))
                channels_group.append(int(round(inplanes * width_mult)))
            else:
                in_channels_group.append(int(round(inplanes * width_mult)))
                inplanes += final_ch / (self.depth // 3 * 1.0)
                channels_group.append(int(round(inplanes * width_mult)))

        conv_bn_swish(
            features,
            3,
            int(round(stem_channel * width_mult)),
            kernel=3,
            stride=2,
            pad=1)

        for block_idx, (in_c, c, t, s, se) in enumerate(
                zip(in_channels_group, channels_group, ts, strides, use_ses)):
            # print('in_c, c, t, s, se:', in_c, c, t, s, se)
            features.append(
                LinearBottleneck(
                    in_channels=in_c,
                    channels=c,
                    t=t,
                    stride=s,
                    use_se=se,
                    se_ratio=se_ratio))

        pen_channels = int(1280 * width_mult)
        conv_bn_swish(features, c, pen_channels)

        features.append(nn.AdaptiveAvgPool2d(1, data_format='channels_first'))
        self.features = nn.Sequential(*features)
        self.output = nn.Sequential(
            nn.Dropout(dropout_ratio),
            ######################################################################################################################################################################################################################################################################################################################
            nn.Conv2d(
                in_channels=pen_channels, out_channels=class_num, kernel_size=1,
                data_format='channels_first'
            ))
        # nn.Conv2D(
        #     pen_channels, class_num, 1, bias_attr=True))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x).squeeze(axis=-1).squeeze(axis=-1)
        return x


# def restore_model(param, model):
#     from tensorlayerx.files import assign_weights
#     tlx2pd_namelast = {'filters': 'weight',  # conv2d
#                        'biases': 'bias',  # linear
#                        'weights': 'weight',  # linear
#                        'gamma': 'weight',  # bn
#                        'beta': 'bias',  # bn
#                        'moving_mean': '_mean',  # bn
#                        'moving_var': '_variance',  # bn
#                        }
#     # print([{i: k} for i, k in model.named_parameters()])
#     model_state = [i for i, k in model.named_parameters()]
#     # for i, k in model.named_parameters():
#     #     print(i)
#     # exit()
#     weights = []
#
#     for i in range(len(model_state)):
#         model_key = model_state[i]
#         model_key_s, model_key_e = model_key.rsplit('.', 1)
#         # print(model_key_s, model_key_e)
#         if model_key_e in tlx2pd_namelast:
#             new_model_state = model_key_s + '.' + tlx2pd_namelast[model_key_e]
#             weights.append(param[new_model_state])
#         else:
#             print('**' * 10, model_key)
#     assign_weights(weights, model)
#     del weights


def _rexnet(arch, pretrained, **kwargs):
    model = ReXNetV1(width_mult=1.0, **kwargs)
    # print(paddle.summary(model, [(1, 3, 256, 256)]))
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch])
        param = paddle.load(weight_path)
        restore_model(param, model)
        # model.set_dict(param)
    return model


def rexnet(pretrained=False, **kwargs):
    return _rexnet('rexnet', pretrained, **kwargs)


if __name__ == '__main__':
    from PIL import Image
    from paddle import to_tensor
    import tensorlayerx as tlx


    def load_image(image_path, mode=None):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = np.array(img).astype(np.float32)
        if mode == "tlx":
            img = np.expand_dims(img, 0)
            img = img / 255.0
            img = tlx.convert_to_tensor(img)
            img = tlx.ops.nhwc_to_nchw(img)
        elif mode == "pd":
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, 0)
            img = img / 255.0
            img = to_tensor(img)
        elif mode == "pt":
            pass
        else:
            img = img.transpose((2, 0, 1))
            img = img / 255.0
        return img


    image_file = "../../images/dog.jpeg"
    model_tlx = rexnet(pretrained=True)
    model_tlx.set_eval()
    img = load_image(image_file, "tlx")
    result_tlx = model_tlx(img)

    file_path = '../../images/imagenet_classes.txt'
    with open(file_path) as f:
        classes = [line.strip() for line in f.readlines()]
    print(classes[np.argmax(result_tlx[0])])
