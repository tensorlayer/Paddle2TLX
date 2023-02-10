from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
from core.workspace import register
from core.workspace import create
from .meta_arch import BaseArch
__all__ = ['SSD']


@register
class SSD(BaseArch):
    """
    Single Shot MultiBox Detector, see https://arxiv.org/abs/1512.02325

    Args:
        backbone (nn.Layer): backbone instance
        ssd_head (nn.Layer): `SSDHead` instance
        post_process (object): `BBoxPostProcess` instance
    """
    __category__ = 'architecture'
    __inject__ = ['post_process']

    def __init__(self, backbone, ssd_head, post_process, r34_backbone=False):
        super(SSD, self).__init__()
        self.backbone = backbone
        self.ssd_head = ssd_head
        self.post_process = post_process
        self.r34_backbone = r34_backbone
        if self.r34_backbone:
            from models.backbones.resnet import ResNet
            assert isinstance(self.backbone, ResNet
                ) and self.backbone.depth == 34, 'If you set r34_backbone=True, please use ResNet-34 as backbone.'
            self.backbone.res_layers[2].blocks[0].branch2a.conv._stride = [1, 1
                ]
            self.backbone.res_layers[2].blocks[0].short.conv._stride = [1, 1]

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        ssd_head = create(cfg['ssd_head'], **kwargs)
        return {'backbone': backbone, 'ssd_head': ssd_head}

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        if self.training:
            return self.ssd_head(body_feats, self.inputs['image'], self.
                inputs['gt_bbox'], self.inputs['gt_class'])
        else:
            preds, anchors = self.ssd_head(body_feats, self.inputs['image'])
            bbox, bbox_num = self.post_process(preds, anchors, self.inputs[
                'im_shape'], self.inputs['scale_factor'])
            return bbox, bbox_num

    def get_loss(self):
        return {'loss': self._forward()}

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
        return output
