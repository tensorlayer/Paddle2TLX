from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from core.workspace import register
from models.bbox_utils import decode_yolo
from models.bbox_utils import xywh2xyxy
from models.bbox_utils import batch_iou_similarity
__all__ = ['YOLOv3Loss']


def bbox_transform(pbox, anchor, downsample):
    pbox = decode_yolo(pbox, anchor, downsample)
    pbox = xywh2xyxy(pbox)
    return pbox


@register
class YOLOv3Loss(nn.Module):
    __inject__ = ['iou_loss', 'iou_aware_loss']
    __shared__ = ['num_classes']

    def __init__(self, num_classes=80, ignore_thresh=0.7, label_smooth=\
        False, downsample=[32, 16, 8], scale_x_y=1.0, iou_loss=None,
        iou_aware_loss=None):
        """
        YOLOv3Loss layer

        Args:
            num_calsses (int): number of foreground classes
            ignore_thresh (float): threshold to ignore confidence loss
            label_smooth (bool): whether to use label smoothing
            downsample (list): downsample ratio for each detection block
            scale_x_y (float): scale_x_y factor
            iou_loss (object): IoULoss instance
            iou_aware_loss (object): IouAwareLoss instance  
        """
        super(YOLOv3Loss, self).__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.label_smooth = label_smooth
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.iou_loss = iou_loss
        self.iou_aware_loss = iou_aware_loss
        self.distill_pairs = []

    def obj_loss(self, pbox, gbox, pobj, tobj, anchor, downsample):
        pbox = decode_yolo(pbox, anchor, downsample)
        pbox = xywh2xyxy(pbox)
        pbox = tensorlayerx.concat(pbox, axis=-1)
        b = pbox.shape[0]
        pbox = pbox.reshape((b, -1, 4))
        gxy = gbox[:, :, 0:2] - gbox[:, :, 2:4] * 0.5
        gwh = gbox[:, :, 0:2] + gbox[:, :, 2:4] * 0.5
        gbox = tensorlayerx.concat([gxy, gwh], axis=-1)
        iou = batch_iou_similarity(pbox, gbox)
        iou.stop_gradient = True
        iou_max = iou.max(2)
        iou_mask = tensorlayerx.cast(iou_max <= self.ignore_thresh, dtype=\
            pbox.dtype)
        iou_mask.stop_gradient = True
        pobj = pobj.reshape((b, -1))
        tobj = tobj.reshape((b, -1))
        obj_mask = tensorlayerx.cast(tobj > 0, dtype=pbox.dtype)
        obj_mask.stop_gradient = True
        loss_obj = tensorlayerx.losses.sigmoid_cross_entropy(pobj, obj_mask,
            reduction='none')
        loss_obj_pos = loss_obj * tobj
        loss_obj_neg = loss_obj * (1 - obj_mask) * iou_mask
        return loss_obj_pos + loss_obj_neg

    def cls_loss(self, pcls, tcls):
        if self.label_smooth:
            delta = min(1.0 / self.num_classes, 1.0 / 40)
            pos, neg = 1 - delta, delta
            aa = tensorlayerx.cast(tcls > 0.0, dtype=tcls.dtype)
            bb = tensorlayerx.cast(tcls <= 0.0, dtype=tcls.dtype)
            tcls = pos * aa + neg * bb
        loss_cls = tensorlayerx.losses.sigmoid_cross_entropy(pcls, tcls,
            reduction='none')
        return loss_cls

    def yolov3_loss(self, p, t, gt_box, anchor, downsample, scale=1.0, eps=\
        1e-10):
        na = len(anchor)
        b, c, h, w = p.shape
        if self.iou_aware_loss:
            ioup, p = p[:, 0:na, :, :], p[:, na:, :, :]
            ioup = ioup.unsqueeze(-1)
        p = p.reshape((b, na, -1, h, w)).transpose((0, 1, 3, 4, 2))
        x, y = p[:, :, :, :, 0:1], p[:, :, :, :, 1:2]
        w, h = p[:, :, :, :, 2:3], p[:, :, :, :, 3:4]
        obj, pcls = p[:, :, :, :, 4:5], p[:, :, :, :, 5:]
        self.distill_pairs.append([x, y, w, h, obj, pcls])
        t = t.transpose((0, 1, 3, 4, 2))
        tx, ty = t[:, :, :, :, 0:1], t[:, :, :, :, 1:2]
        tw, th = t[:, :, :, :, 2:3], t[:, :, :, :, 3:4]
        tscale = t[:, :, :, :, 4:5]
        tobj, tcls = t[:, :, :, :, 5:6], t[:, :, :, :, 6:]
        tscale_obj = tscale * tobj
        loss = dict()
        sx = tensorlayerx.ops.sigmoid(x)
        sy = tensorlayerx.ops.sigmoid(y)
        x = scale * sx - 0.5 * (scale - 1.0)
        y = scale * sy - 0.5 * (scale - 1.0)
        if abs(scale - 1.0) < eps:
            loss_x = tensorlayerx.losses.binary_cross_entropy(x, tx,
                reduction='none')
            loss_y = tensorlayerx.losses.binary_cross_entropy(y, ty,
                reduction='none')
            loss_xy = tscale_obj * (loss_x + loss_y)
        else:
            loss_x = tensorlayerx.ops.abs(x - tx)
            loss_y = tensorlayerx.ops.abs(y - ty)
            loss_xy = tscale_obj * (loss_x + loss_y)
        loss_xy = loss_xy.sum([1, 2, 3, 4]).mean()
        loss_w = tensorlayerx.ops.abs(w - tw)
        loss_h = tensorlayerx.ops.abs(h - th)
        loss_wh = tscale_obj * (loss_w + loss_h)
        loss_wh = loss_wh.sum([1, 2, 3, 4]).mean()
        loss['loss_xy'] = loss_xy
        loss['loss_wh'] = loss_wh
        if self.iou_loss is not None:
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou = self.iou_loss(pbox, gbox)
            loss_iou = loss_iou * tscale_obj
            loss_iou = loss_iou.sum([1, 2, 3, 4]).mean()
            loss['loss_iou'] = loss_iou
        if self.iou_aware_loss is not None:
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou_aware = self.iou_aware_loss(ioup, pbox, gbox)
            loss_iou_aware = loss_iou_aware * tobj
            loss_iou_aware = loss_iou_aware.sum([1, 2, 3, 4]).mean()
            loss['loss_iou_aware'] = loss_iou_aware
        box = [x, y, w, h]
        loss_obj = self.obj_loss(box, gt_box, obj, tobj, anchor, downsample)
        loss_obj = loss_obj.sum(-1).mean()
        loss['loss_obj'] = loss_obj
        loss_cls = self.cls_loss(pcls, tcls) * tobj
        loss_cls = loss_cls.sum([1, 2, 3, 4]).mean()
        loss['loss_cls'] = loss_cls
        return loss

    def forward(self, inputs, targets, anchors):
        np = len(inputs)
        gt_targets = [targets['target{}'.format(i)] for i in range(np)]
        gt_box = targets['gt_bbox']
        yolo_losses = dict()
        self.distill_pairs.clear()
        for x, t, anchor, downsample in zip(inputs, gt_targets, anchors,
            self.downsample):
            yolo_loss = self.yolov3_loss(x, t, gt_box, anchor, downsample,
                self.scale_x_y)
            for k, v in yolo_loss.items():
                if k in yolo_losses:
                    yolo_losses[k] += v
                else:
                    yolo_losses[k] = v
        loss = 0
        for k, v in yolo_losses.items():
            loss += v
        yolo_losses['loss'] = loss
        return yolo_losses
