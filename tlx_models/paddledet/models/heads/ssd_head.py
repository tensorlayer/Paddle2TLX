import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from core.workspace import register
from paddle.regularizer import L2Decay
from tensorlayerx.nn.initializers import xavier_uniform
from ..layers import AnchorGeneratorSSD
from ..cls_utils import _get_class_default_kwargs


class SepConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
        conv_decay=0.0):
        super(SepConvLayer, self).__init__()
        self.dw_conv = nn.GroupConv2d(in_channels=in_channels, out_channels
            =in_channels, kernel_size=kernel_size, stride=1, padding=\
            padding, W_init=xavier_uniform(), b_init=False, n_group=\
            in_channels, data_format='channels_first')
        self.bn = nn.BatchNorm2d(num_features=in_channels, data_format=\
            'channels_first')
        self.pw_conv = nn.GroupConv2d(in_channels=in_channels, out_channels
            =out_channels, kernel_size=1, stride=1, padding=0, W_init=\
            xavier_uniform(), b_init=False, data_format='channels_first')

    def forward(self, x):
        x = self.dw_conv(x)
        x = tensorlayerx.nn.ReLU6()(self.bn(x))
        x = self.pw_conv(x)
        return x


class SSDExtraHead(nn.Module):

    def __init__(self, in_channels=256, out_channels=([256, 512], [256, 512
        ], [128, 256], [128, 256], [128, 256]), strides=(2, 2, 2, 1, 1),
        paddings=(1, 1, 1, 0, 0)):
        super(SSDExtraHead, self).__init__()
        self.convs = nn.ModuleList()
        for out_channel, stride, padding in zip(out_channels, strides, paddings
            ):
            self.convs.append(self._make_layers(in_channels, out_channel[0],
                out_channel[1], stride, padding))
            in_channels = out_channel[-1]

    def _make_layers(self, c_in, c_hidden, c_out, stride_3x3, padding_3x3):
        return nn.Sequential([nn.GroupConv2d(in_channels=c_in, out_channels
            =c_hidden, kernel_size=1, padding=0, data_format=\
            'channels_first'), nn.ReLU(), nn.GroupConv2d(in_channels=\
            c_hidden, out_channels=c_out, kernel_size=3, stride=stride_3x3,
            padding=padding_3x3, data_format='channels_first'), nn.ReLU()])

    def forward(self, x):
        out = [x]
        for conv_layer in self.convs:
            out.append(conv_layer(out[-1]))
        return out


@register
class SSDHead(nn.Module):
    """
    SSDHead

    Args:
        num_classes (int): Number of classes
        in_channels (list): Number of channels per input feature
        anchor_generator (dict): Configuration of 'AnchorGeneratorSSD' instance
        kernel_size (int): Conv kernel size
        padding (int): Conv padding
        use_sepconv (bool): Use SepConvLayer if true
        conv_decay (float): Conv regularization coeff
        loss (object): 'SSDLoss' instance
        use_extra_head (bool): If use ResNet34 as baskbone, you should set `use_extra_head`=True
    """
    __shared__ = ['num_classes']
    __inject__ = ['anchor_generator', 'loss']

    def __init__(self, num_classes=80, in_channels=(512, 1024, 512, 256, 
        256, 256), anchor_generator=_get_class_default_kwargs(
        AnchorGeneratorSSD), kernel_size=3, padding=1, use_sepconv=False,
        conv_decay=0.0, loss='SSDLoss', use_extra_head=False):
        super(SSDHead, self).__init__()
        self.num_classes = num_classes + 1
        self.in_channels = in_channels
        self.anchor_generator = anchor_generator
        self.loss = loss
        self.use_extra_head = use_extra_head
        if self.use_extra_head:
            self.ssd_extra_head = SSDExtraHead()
            self.in_channels = [256, 512, 512, 256, 256, 256]
        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGeneratorSSD(**anchor_generator)
        self.num_priors = self.anchor_generator.num_priors
        self.box_convs = []
        self.score_convs = []
        for i, num_prior in enumerate(self.num_priors):
            box_conv_name = 'boxes{}'.format(i)
            if not use_sepconv:
                box_conv = self.add_sublayer(box_conv_name, nn.GroupConv2d(
                    in_channels=self.in_channels[i], out_channels=num_prior *
                    4, kernel_size=kernel_size, padding=padding,
                    data_format='channels_first'))
            else:
                box_conv = self.add_sublayer(box_conv_name, SepConvLayer(
                    in_channels=self.in_channels[i], out_channels=num_prior *
                    4, kernel_size=kernel_size, padding=padding, conv_decay
                    =conv_decay))
            self.box_convs.append(box_conv)
            score_conv_name = 'scores{}'.format(i)
            if not use_sepconv:
                score_conv = self.add_sublayer(score_conv_name, nn.
                    GroupConv2d(in_channels=self.in_channels[i],
                    out_channels=num_prior * self.num_classes, kernel_size=\
                    kernel_size, padding=padding, data_format='channels_first')
                    )
            else:
                score_conv = self.add_sublayer(score_conv_name,
                    SepConvLayer(in_channels=self.in_channels[i],
                    out_channels=num_prior * self.num_classes, kernel_size=\
                    kernel_size, padding=padding, conv_decay=conv_decay))
            self.score_convs.append(score_conv)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape]}

    def forward(self, feats, image, gt_bbox=None, gt_class=None):
        if self.use_extra_head:
            assert len(feats
                ) == 1, 'If you set use_extra_head=True, backbone feature list length should be 1.'
            feats = self.ssd_extra_head(feats[0])
        box_preds = []
        cls_scores = []
        for feat, box_conv, score_conv in zip(feats, self.box_convs, self.
            score_convs):
            box_pred = box_conv(feat)
            box_pred = tensorlayerx.transpose(box_pred, [0, 2, 3, 1])
            box_pred = tensorlayerx.reshape(box_pred, [0, -1, 4])
            box_preds.append(box_pred)
            cls_score = score_conv(feat)
            cls_score = tensorlayerx.transpose(cls_score, [0, 2, 3, 1])
            cls_score = tensorlayerx.reshape(cls_score, [0, -1, self.
                num_classes])
            cls_scores.append(cls_score)
        prior_boxes = self.anchor_generator(feats, image)
        if self.training:
            return self.get_loss(box_preds, cls_scores, gt_bbox, gt_class,
                prior_boxes)
        else:
            return (box_preds, cls_scores), prior_boxes

    def get_loss(self, boxes, scores, gt_bbox, gt_class, prior_boxes):
        return self.loss(boxes, scores, gt_bbox, gt_class, prior_boxes)
