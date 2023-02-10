from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
from core.workspace import register
from core.workspace import create
from .meta_arch import BaseArch
import tensorlayerx
__all__ = ['RetinaNet']


@register
class RetinaNet(BaseArch):
    __category__ = 'architecture'

    def __init__(self, backbone, neck, head):
        super(RetinaNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)
        kwargs = {'input_shape': neck.out_shape}
        head = create(cfg['head'], **kwargs)
        return {'backbone': backbone, 'neck': neck, 'head': head}

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats)
        if self.training:
            return self.head(neck_feats, self.inputs)
        else:
            head_outs = self.head(neck_feats)
            bbox, bbox_num = self.head.post_process(head_outs, self.inputs[
                'im_shape'], self.inputs['scale_factor'])
            return {'bbox': bbox, 'bbox_num': bbox_num}

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
