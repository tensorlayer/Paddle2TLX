# dpn68  dpn107

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys

import tensorlayerx as tlx

import paddle

from utils.load_model import restore_model
from utils.download import get_weights_path_from_url

MODEL_URLS = {
    "DPN68":
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN68_pretrained.pdparams",
    "DPN107":
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN107_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


class ConvBNLayer(tlx.nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 pad=0,
                 groups=1,
                 act="relu",
                 name=None):
        super(ConvBNLayer, self).__init__()

        # self._conv = Conv2D(
        #     in_channels=num_channels,
        #     out_channels=num_filters,
        #     kernel_size=filter_size,
        #     stride=stride,
        #     padding=pad,
        #     groups=groups,
        #     weight_attr=ParamAttr(name=name + "_weights"),
        #     bias_attr=False)

        # groups      ============================这个注意
        self._conv = tlx.nn.GroupConv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=pad,
            n_group=groups,
            W_init=tlx.initializers.xavier_uniform(),
            b_init=None,
            data_format='channels_first',)
        # self._batch_norm = BatchNorm(
        #     num_filters,
        #     act=act,
        #     param_attr=ParamAttr(name=name + '_bn_scale'),
        #     bias_attr=ParamAttr(name + '_bn_offset'),
        #     moving_mean_name=name + '_bn_mean',
        #     moving_variance_name=name + '_bn_variance')
        self._batch_norm = tlx.nn.BatchNorm(
            num_features=num_filters,
            act=act,
            data_format='channels_first',
            moving_mean_init=tlx.initializers.xavier_uniform(),
            moving_var_init=tlx.initializers.xavier_uniform()
        )

    def forward(self, input):
        y = self._conv(input)
        y = self._batch_norm(y)
        return y


class BNACConvLayer(tlx.nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 pad=0,
                 groups=1,
                 act="relu",
                 name=None):
        super(BNACConvLayer, self).__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters

        # self._batch_norm = BatchNorm(
        #     num_channels,
        #     act=act,
        #     param_attr=ParamAttr(name=name + '_bn_scale'),
        #     bias_attr=ParamAttr(name + '_bn_offset'),
        #     moving_mean_name=name + '_bn_mean',
        #     moving_variance_name=name + '_bn_variance')
        self._batch_norm = tlx.nn.BatchNorm(
            num_features=num_channels,
            act=act,
            data_format='channels_first',
            moving_mean_init=tlx.initializers.xavier_uniform(),
            moving_var_init=tlx.initializers.xavier_uniform()
        )

        # self._conv = Conv2D(
        #     in_channels=num_channels,
        #     out_channels=num_filters,
        #     kernel_size=filter_size,
        #     stride=stride,
        #     padding=pad,
        #     groups=groups,
        #     weight_attr=ParamAttr(name=name + "_weights"),
        #     bias_attr=False)

        self._conv = tlx.nn.GroupConv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=pad,
            n_group=groups,
            W_init=tlx.initializers.xavier_uniform(),
            b_init=None,
            data_format='channels_first')

    def forward(self, input):
        y = self._batch_norm(input) # [1,10,56,56]
        y = self._conv(y)
        return y


class DualPathFactory(tlx.nn.Module):
    def __init__(self,
                 num_channels,
                 num_1x1_a,
                 num_3x3_b,
                 num_1x1_c,
                 inc,
                 G,
                 _type='normal',
                 name=None):
        super(DualPathFactory, self).__init__()

        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.name = name

        kw = 3
        kh = 3
        pw = (kw - 1) // 2
        ph = (kh - 1) // 2

        # type
        if _type == 'proj':
            key_stride = 1
            self.has_proj = True
        elif _type == 'down':
            key_stride = 2
            self.has_proj = True
        elif _type == 'normal':
            key_stride = 1
            self.has_proj = False
        else:
            print("not implemented now!!!")
            sys.exit(1)

        data_in_ch = sum(num_channels) if isinstance(num_channels,
                                                     list) else num_channels

        if self.has_proj:
            self.c1x1_w_func = BNACConvLayer(
                num_channels=data_in_ch,
                num_filters=num_1x1_c + 2 * inc,
                filter_size=(1, 1),
                pad=(0, 0),
                stride=(key_stride, key_stride),
                name=name + "_match")

        self.c1x1_a_func = BNACConvLayer(
            num_channels=data_in_ch,
            num_filters=num_1x1_a,
            filter_size=(1, 1),
            pad=(0, 0),
            name=name + "_conv1")

        self.c3x3_b_func = BNACConvLayer(
            num_channels=num_1x1_a,
            num_filters=num_3x3_b,
            filter_size=(kw, kh),
            pad=(pw, ph),
            stride=(key_stride, key_stride),
            groups=G,
            name=name + "_conv2")

        self.c1x1_c_func = BNACConvLayer(
            num_channels=num_3x3_b,
            num_filters=num_1x1_c + inc,
            filter_size=(1, 1),
            pad=(0, 0),
            name=name + "_conv3")

    def forward(self, input):
        # PROJ
        if isinstance(input, list):
            data_in = tlx.concat([input[0], input[1]], axis=1)
        else:
            data_in = input

        if self.has_proj:
            c1x1_w = self.c1x1_w_func(data_in)
            data_o1, data_o2 = paddle.split(c1x1_w, num_or_sections=[self.num_1x1_c, 2 * self.inc], axis=1)
        else:
            data_o1 = input[0]
            data_o2 = input[1]

        c1x1_a = self.c1x1_a_func(data_in)
        c3x3_b = self.c3x3_b_func(c1x1_a)
        c1x1_c = self.c1x1_c_func(c3x3_b)

        c1x1_c1, c1x1_c2 = paddle.split(c1x1_c, num_or_sections=[self.num_1x1_c, self.inc], axis=1)

        # OUTPUTS
        summ = tlx.ops.add(data_o1, c1x1_c1)
        dense = tlx.concat([data_o2, c1x1_c2], axis=1)
        # tensor, channels
        return [summ, dense]


class DPN(tlx.nn.Module):
    def __init__(self, layers=68, class_num=1000):
        super(DPN, self).__init__()

        self._class_num = class_num

        args = self.get_net_args(layers)
        bws = args['bw']
        inc_sec = args['inc_sec']
        rs = args['r']
        k_r = args['k_r']
        k_sec = args['k_sec']
        G = args['G']
        init_num_filter = args['init_num_filter']
        init_filter_size = args['init_filter_size']
        init_padding = args['init_padding']

        self.k_sec = k_sec

        self.conv1_x_1_func = ConvBNLayer(
            num_channels=3,
            num_filters=init_num_filter,
            filter_size=init_filter_size,
            stride=2,
            pad=init_padding,
            act='relu',
            name="conv1")

        self.pool2d_max = tlx.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, data_format='channels_first')

        num_channel_dpn = init_num_filter

        self.dpn_func_list = []
        # conv2 - conv5
        match_list, num = [], 0
        for gc in range(4):
            bw = bws[gc]
            inc = inc_sec[gc]
            R = (k_r * bw) // rs[gc]
            if gc == 0:
                _type1 = 'proj'
                _type2 = 'normal'
                match = 1
            else:
                _type1 = 'down'
                _type2 = 'normal'
                match = match + k_sec[gc - 1]
            match_list.append(match)
            self.dpn_func_list.append(
                self.add_sublayer(
                    "dpn{}".format(match),
                    DualPathFactory(
                        num_channels=num_channel_dpn,
                        num_1x1_a=R,
                        num_3x3_b=R,
                        num_1x1_c=bw,
                        inc=inc,
                        G=G,
                        _type=_type1,
                        name="dpn" + str(match))))
            num_channel_dpn = [bw, 3 * inc]

            for i_ly in range(2, k_sec[gc] + 1):
                num += 1
                if num in match_list:
                    num += 1
                self.dpn_func_list.append(
                    self.add_sublayer(
                        "dpn{}".format(num),
                        DualPathFactory(
                            num_channels=num_channel_dpn,
                            num_1x1_a=R,
                            num_3x3_b=R,
                            num_1x1_c=bw,
                            inc=inc,
                            G=G,
                            _type=_type2,
                            name="dpn" + str(num))))

                num_channel_dpn = [
                    num_channel_dpn[0], num_channel_dpn[1] + inc
                ]

        out_channel = sum(num_channel_dpn)

        # self.conv5_x_x_bn = BatchNorm(
        #     num_channels=sum(num_channel_dpn),
        #     act="relu",
        #     param_attr=ParamAttr(name='final_concat_bn_scale'),
        #     bias_attr=ParamAttr('final_concat_bn_offset'),
        #     moving_mean_name='final_concat_bn_mean',
        #     moving_variance_name='final_concat_bn_variance')
        self.conv5_x_x_bn = tlx.nn.BatchNorm(
            num_features=sum(num_channel_dpn),
            act="relu",
            data_format='channels_first',
            moving_mean_init=tlx.initializers.xavier_uniform(),
            moving_var_init=tlx.initializers.xavier_uniform()
        )

        self.pool2d_avg = tlx.nn.AdaptiveAvgPool2d(1, data_format='channels_first')

        stdv = 0.01

        self.out = tlx.nn.Linear(
            in_features=out_channel,
            out_features=class_num,
            W_init=tlx.initializers.random_uniform(minval=-stdv, maxval=stdv),
            b_init=tlx.initializers.xavier_uniform())

    def get_net_args(self, layers):
        if layers == 68:
            k_r = 128
            G = 32
            k_sec = [3, 4, 12, 3]
            inc_sec = [16, 32, 32, 64]
            bw = [64, 128, 256, 512]
            r = [64, 64, 64, 64]
            init_num_filter = 10
            init_filter_size = 3
            init_padding = 1
        elif layers == 107:
            k_r = 200
            G = 50
            k_sec = [4, 8, 20, 3]
            inc_sec = [20, 64, 64, 128]
            bw = [256, 512, 1024, 2048]
            r = [256, 256, 256, 256]
            init_num_filter = 128
            init_filter_size = 7
            init_padding = 3
        else:
            raise NotImplementedError
        net_arg = {
            'k_r': k_r,
            'G': G,
            'k_sec': k_sec,
            'inc_sec': inc_sec,
            'bw': bw,
            'r': r
        }
        net_arg['init_num_filter'] = init_num_filter
        net_arg['init_filter_size'] = init_filter_size
        net_arg['init_padding'] = init_padding

        return net_arg

    def forward(self, input):
        conv1_x_1 = self.conv1_x_1_func(input)
        convX_x_x = self.pool2d_max(conv1_x_1)

        dpn_idx = 0
        for gc in range(4):
            convX_x_x = self.dpn_func_list[dpn_idx](convX_x_x)
            dpn_idx += 1
            for i_ly in range(2, self.k_sec[gc] + 1):
                convX_x_x = self.dpn_func_list[dpn_idx](convX_x_x)

                dpn_idx += 1

        conv5_x_x = tlx.concat(convX_x_x, axis=1)
        conv5_x_x = self.conv5_x_x_bn(conv5_x_x)

        y = self.pool2d_avg(conv5_x_x)
        y = tlx.nn.Flatten(name='flatten')(y)
        y = self.out(y)
        return y


def _dpn68(arch, pretrained, **kwargs):
    model = DPN(layers=68, **kwargs)
    if pretrained:
        assert arch in MODEL_URLS, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(MODEL_URLS[arch])

        param = paddle.load(weight_path)
        restore_model(param, model)

    return model


def _dpn107(arch, pretrained, **kwargs):
    model = DPN(layers=107, **kwargs)
    if pretrained:
        assert arch in MODEL_URLS, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(MODEL_URLS[arch])

        param = paddle.load(weight_path)
        restore_model(param, model)

    return model


def dpn68(pretrained=False, **kwargs):
    return _dpn68("DPN68", pretrained, **kwargs)

def dpn107(pretrained=False, **kwargs):
    return _dpn107("DPN107", pretrained, **kwargs)
