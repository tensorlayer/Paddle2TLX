from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
from core.workspace import register
from core.workspace import create
from .meta_arch import BaseArch
__all__ = ['GFL']


@register
class GFL(BaseArch):
    """
    Generalized Focal Loss network, see https://arxiv.org/abs/2006.04388

    Args:
        backbone (object): backbone instance
        neck (object): 'FPN' instance
        head (object): 'GFLHead' instance
    """
    __category__ = 'architecture'

    def __init__(self, backbone, neck, head='GFLHead'):
        super(GFL, self).__init__()
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
            im_shape = self.inputs['im_shape']
            scale_factor = self.inputs['scale_factor']
            bboxes, bbox_num = self.head.post_process(head_outs, im_shape,
                scale_factor)
            return bboxes, bbox_num
        else:
            return head_outs

    def get_loss(self):
        loss = {}
        head_outs = self._forward()
        loss_gfl = self.head.get_loss(head_outs, self.inputs)
        loss.update(loss_gfl)
        total_loss = tensorlayerx.ops.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
        return output
