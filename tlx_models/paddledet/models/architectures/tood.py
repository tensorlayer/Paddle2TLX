from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
from core.workspace import register
from core.workspace import create
from .meta_arch import BaseArch
__all__ = ['TOOD']


@register
class TOOD(BaseArch):
    """
    TOOD: Task-aligned One-stage Object Detection, see https://arxiv.org/abs/2108.07755
    Args:
        backbone (nn.Layer): backbone instance
        neck (nn.Layer): 'FPN' instance
        head (nn.Layer): 'TOODHead' instance
    """
    __category__ = 'architecture'

    def __init__(self, backbone, neck, head):
        super(TOOD, self).__init__()
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
        fpn_feats = self.neck(body_feats)
        head_outs = self.head(fpn_feats)
        if not self.training:
            bboxes, bbox_num = self.head.post_process(head_outs, self.
                inputs['im_shape'], self.inputs['scale_factor'])
            return bboxes, bbox_num
        else:
            loss = self.head.get_loss(head_outs, self.inputs)
            return loss

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
        return output
