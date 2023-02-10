import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import xavier_uniform
from paddle.regularizer import L2Decay
from core.workspace import register
from core.workspace import serializable
from ..shape_spec import ShapeSpec
from ..backbones.lcnet import DepthwiseSeparable
from .csp_pan import ConvBNLayer
from .csp_pan import Channel_T
from .csp_pan import DPModule
__all__ = ['LCPAN']


@register
@serializable
class LCPAN(nn.Module):
    """Path Aggregation Network with LCNet module.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        kernel_size (int): The conv2d kernel size of this Module.
        num_features (int): Number of output features of CSPPAN module.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 1
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: True
    """

    def __init__(self, in_channels, out_channels, kernel_size=5,
        num_features=3, use_depthwise=True, act='hard_swish',
        spatial_scales=[0.125, 0.0625, 0.03125]):
        super(LCPAN, self).__init__()
        self.conv_t = Channel_T(in_channels, out_channels, act=act)
        in_channels = [out_channels] * len(spatial_scales)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_scales = spatial_scales
        self.num_features = num_features
        conv_func = DPModule if use_depthwise else ConvBNLayer
        NET_CONFIG = {'block1': [[kernel_size, out_channels * 2, 
            out_channels * 2, 1, False], [kernel_size, out_channels * 2,
            out_channels, 1, False]], 'block2': [[kernel_size, out_channels *
            2, out_channels * 2, 1, False], [kernel_size, out_channels * 2,
            out_channels, 1, False]]}
        if self.num_features == 4:
            self.first_top_conv = conv_func(in_channels[0], in_channels[0],
                kernel_size, stride=2, act=act)
            self.second_top_conv = conv_func(in_channels[0], in_channels[0],
                kernel_size, stride=2, act=act)
            self.spatial_scales.append(self.spatial_scales[-1] / 2)
        self.upsample = paddle2tlx.pd2tlx.ops.tlxops.tlx_Upsample(scale_factor
            =2, mode='nearest', data_format='channels_first')
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.top_down_blocks.append(nn.Sequential([*[DepthwiseSeparable
                (num_channels=in_c, num_filters=out_c, dw_size=k, stride=s,
                use_se=se) for i, (k, in_c, out_c, s, se) in enumerate(
                NET_CONFIG['block1'])]]))
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(conv_func(in_channels[idx], in_channels
                [idx], kernel_size=kernel_size, stride=2, act=act))
            self.bottom_up_blocks.append(nn.Sequential([*[
                DepthwiseSeparable(num_channels=in_c, num_filters=out_c,
                dw_size=k, stride=s, use_se=se) for i, (k, in_c, out_c, s,
                se) in enumerate(NET_CONFIG['block2'])]]))

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.
        Returns:
            tuple[Tensor]: CSPPAN features.
        """
        assert len(inputs) == len(self.in_channels)
        inputs = self.conv_t(inputs)
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            upsample_feat = self.upsample(feat_heigh)
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                tensorlayerx.concat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](tensorlayerx.concat([
                downsample_feat, feat_height], 1))
            outs.append(out)
        top_features = None
        if self.num_features == 4:
            top_features = self.first_top_conv(inputs[-1])
            top_features = top_features + self.second_top_conv(outs[-1])
            outs.append(top_features)
        return tuple(outs)

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.out_channels, stride=1.0 / s) for s in
            self.spatial_scales]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape]}
