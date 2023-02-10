import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import xavier_uniform
from tensorlayerx.nn.initializers import XavierUniform
from core.workspace import register
from core.workspace import serializable
from models.layers import ConvNormLayer
from ..shape_spec import ShapeSpec
__all__ = ['FPN']


@register
@serializable
class FPN(nn.Module):
    """
    Feature Pyramid Network, see https://arxiv.org/abs/1612.03144

    Args:
        in_channels (list[int]): input channels of each level which can be 
            derived from the output shape of backbone by from_config
        out_channel (int): output channel of each level
        spatial_scales (list[float]): the spatial scales between input feature
            maps and original input image which can be derived from the output 
            shape of backbone by from_config
        has_extra_convs (bool): whether to add extra conv to the last level.
            default False
        extra_stage (int): the number of extra stages added to the last level.
            default 1
        use_c5 (bool): Whether to use c5 as the input of extra stage, 
            otherwise p5 is used. default True
        norm_type (string|None): The normalization type in FPN module. If 
            norm_type is None, norm will not be used after conv and if 
            norm_type is string, bn, gn, sync_bn are available. default None
        norm_decay (float): weight decay for normalization layer weights.
            default 0.
        freeze_norm (bool): whether to freeze normalization layer.  
            default False
        relu_before_extra_convs (bool): whether to add relu before extra convs.
            default False
        
    """

    def __init__(self, in_channels, out_channel, spatial_scales=[0.25, 
        0.125, 0.0625, 0.03125], has_extra_convs=False, extra_stage=1,
        use_c5=True, norm_type=None, norm_decay=0.0, freeze_norm=False,
        relu_before_extra_convs=True):
        super(FPN, self).__init__()
        self.out_channel = out_channel
        for s in range(extra_stage):
            spatial_scales = spatial_scales + [spatial_scales[-1] / 2.0]
        self.spatial_scales = spatial_scales
        self.has_extra_convs = has_extra_convs
        self.extra_stage = extra_stage
        self.use_c5 = use_c5
        self.relu_before_extra_convs = relu_before_extra_convs
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self.lateral_convs = []
        self.fpn_convs = []
        fan = out_channel * 3 * 3
        st_stage = 4 - len(in_channels)
        ed_stage = st_stage + len(in_channels) - 1
        for i in range(st_stage, ed_stage + 1):
            if i == 3:
                lateral_name = 'fpn_inner_res5_sum'
            else:
                lateral_name = 'fpn_inner_res{}_sum_lateral'.format(i + 2)
            in_c = in_channels[i - st_stage]
            if self.norm_type is not None:
                lateral = self.add_sublayer(lateral_name, ConvNormLayer(
                    ch_in=in_c, ch_out=out_channel, filter_size=1, stride=1,
                    norm_type=self.norm_type, norm_decay=self.norm_decay,
                    freeze_norm=self.freeze_norm, initializer=XavierUniform
                    (fan_out=in_c)))
            else:
                lateral = self.add_sublayer(lateral_name, nn.GroupConv2d(
                    in_channels=in_c, out_channels=out_channel, kernel_size
                    =1, W_init=xavier_uniform(), padding=0, data_format=\
                    'channels_first'))
            self.lateral_convs.append(lateral)
            fpn_name = 'fpn_res{}_sum'.format(i + 2)
            if self.norm_type is not None:
                fpn_conv = self.add_sublayer(fpn_name, ConvNormLayer(ch_in=\
                    out_channel, ch_out=out_channel, filter_size=3, stride=\
                    1, norm_type=self.norm_type, norm_decay=self.norm_decay,
                    freeze_norm=self.freeze_norm, initializer=XavierUniform
                    (fan_out=fan)))
            else:
                fpn_conv = self.add_sublayer(fpn_name, nn.GroupConv2d(
                    in_channels=out_channel, out_channels=out_channel,
                    kernel_size=3, padding=1, W_init=xavier_uniform(),
                    data_format='channels_first'))
            self.fpn_convs.append(fpn_conv)
        if self.has_extra_convs:
            for i in range(self.extra_stage):
                lvl = ed_stage + 1 + i
                if i == 0 and self.use_c5:
                    in_c = in_channels[-1]
                else:
                    in_c = out_channel
                extra_fpn_name = 'fpn_{}'.format(lvl + 2)
                if self.norm_type is not None:
                    extra_fpn_conv = self.add_sublayer(extra_fpn_name,
                        ConvNormLayer(ch_in=in_c, ch_out=out_channel,
                        filter_size=3, stride=2, norm_type=self.norm_type,
                        norm_decay=self.norm_decay, freeze_norm=self.
                        freeze_norm, initializer=XavierUniform(fan_out=fan)))
                else:
                    extra_fpn_conv = self.add_sublayer(extra_fpn_name, nn.
                        GroupConv2d(in_channels=in_c, out_channels=\
                        out_channel, kernel_size=3, stride=2, padding=1,
                        W_init=xavier_uniform(), data_format='channels_first'))
                self.fpn_convs.append(extra_fpn_conv)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape],
            'spatial_scales': [(1.0 / i.stride) for i in input_shape]}

    def forward(self, body_feats):
        laterals = []
        num_levels = len(body_feats)
        for i in range(num_levels):
            laterals.append(self.lateral_convs[i](body_feats[i]))
        for i in range(1, num_levels):
            lvl = num_levels - i
            upsample = paddle.nn.functional.interpolate(laterals[lvl],
                scale_factor=2.0, mode='nearest')
            laterals[lvl - 1] += upsample
        fpn_output = []
        for lvl in range(num_levels):
            fpn_output.append(self.fpn_convs[lvl](laterals[lvl]))
        if self.extra_stage > 0:
            if not self.has_extra_convs:
                assert self.extra_stage == 1, 'extra_stage should be 1 if FPN has not extra convs'
                fpn_output.append(paddle.nn.functional.max_pool2d(
                    fpn_output[-1], 1, stride=2))
            else:
                if self.use_c5:
                    extra_source = body_feats[-1]
                else:
                    extra_source = fpn_output[-1]
                fpn_output.append(self.fpn_convs[num_levels](extra_source))
                for i in range(1, self.extra_stage):
                    if self.relu_before_extra_convs:
                        fpn_output.append(self.fpn_convs[num_levels + i](
                            tensorlayerx.ops.relu(fpn_output[-1])))
                    else:
                        fpn_output.append(self.fpn_convs[num_levels + i](
                            fpn_output[-1]))
        return fpn_output

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.out_channel, stride=1.0 / s) for s in
            self.spatial_scales]
