from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
from core.workspace import register
from core.workspace import create
from .meta_arch import BaseArch
__all__ = ['YOLOv3']


@register
class YOLOv3(BaseArch):
    __category__ = 'architecture'
    __shared__ = ['data_format']
    __inject__ = ['post_process']

    def __init__(self, backbone='DarkNet', neck='YOLOv3FPN', yolo_head=\
        'YOLOv3Head', post_process='BBoxPostProcess', data_format=\
        'channels_first', for_mot=False):
        """
        YOLOv3 network, see https://arxiv.org/abs/1804.02767

        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck instance
            yolo_head (nn.Layer): anchor_head instance
            bbox_post_process (object): `BBoxPostProcess` instance
            data_format (str): data format, NCHW or NHWC
            for_mot (bool): whether return other features for multi-object tracking
                models, default False in pure object detection models.
        """
        super(YOLOv3, self).__init__(data_format=data_format)
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head
        self.post_process = post_process
        self.for_mot = for_mot
        self.return_idx = False

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)
        kwargs = {'input_shape': neck.out_shape}
        yolo_head = create(cfg['yolo_head'], **kwargs)
        return {'backbone': backbone, 'neck': neck, 'yolo_head': yolo_head}

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats, self.for_mot)
        if isinstance(neck_feats, dict):
            assert self.for_mot == True
            emb_feats = neck_feats['emb_feats']
            neck_feats = neck_feats['yolo_feats']
        if self.training:
            yolo_losses = self.yolo_head(neck_feats, self.inputs)
            if self.for_mot:
                return {'det_losses': yolo_losses, 'emb_feats': emb_feats}
            else:
                return yolo_losses
        else:
            yolo_head_outs = self.yolo_head(neck_feats)
            if self.for_mot:
                boxes_idx, bbox, bbox_num, nms_keep_idx = self.post_process(
                    yolo_head_outs, self.yolo_head.mask_anchors)
                output = {'bbox': bbox, 'bbox_num': bbox_num, 'boxes_idx':
                    boxes_idx, 'nms_keep_idx': nms_keep_idx, 'emb_feats':
                    emb_feats}
            else:
                if self.return_idx:
                    _, bbox, bbox_num, _ = self.post_process(yolo_head_outs,
                        self.yolo_head.mask_anchors)
                elif self.post_process is not None:
                    bbox, bbox_num = self.post_process(yolo_head_outs, self
                        .yolo_head.mask_anchors, self.inputs['im_shape'],
                        self.inputs['scale_factor'])
                else:
                    bbox, bbox_num = self.yolo_head.post_process(yolo_head_outs
                        , self.inputs['scale_factor'])
                output = {'bbox': bbox, 'bbox_num': bbox_num}
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
