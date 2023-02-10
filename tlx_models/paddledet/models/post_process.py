import tensorlayerx as tlx
import paddle
import paddle2tlx
import numpy as np
import tensorlayerx
from core.workspace import register
from models.bbox_utils import nonempty_bbox
from models.layers import TTFBox
__all__ = ['BBoxPostProcess', 'MaskPostProcess', 'FCOSPostProcess',
    'CenterNetPostProcess', 'SparsePostProcess']


@register
class BBoxPostProcess(object):
    __shared__ = ['num_classes', 'export_onnx', 'export_eb']
    __inject__ = ['decode', 'nms']

    def __init__(self, num_classes=80, decode=None, nms=None, export_onnx=\
        False, export_eb=False):
        super(BBoxPostProcess, self).__init__()
        self.num_classes = num_classes
        self.decode = decode
        self.nms = nms
        self.export_onnx = export_onnx
        self.export_eb = export_eb

    def __call__(self, head_out, rois, im_shape, scale_factor):
        """
        Decode the bbox and do NMS if needed.

        Args:
            head_out (tuple): bbox_pred and cls_prob of bbox_head output.
            rois (tuple): roi and rois_num of rpn_head output.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
            export_onnx (bool): whether export model to onnx
        Returns:
            bbox_pred (Tensor): The output prediction with shape [N, 6], including
                labels, scores and bboxes. The size of bboxes are corresponding
                to the input image, the bboxes may be used in other branch.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
        """
        if self.nms is not None:
            bboxes, score = self.decode(head_out, rois, im_shape, scale_factor)
            bbox_pred, bbox_num, _ = self.nms(bboxes, score, self.num_classes)
        else:
            bbox_pred, bbox_num = self.decode(head_out, rois, im_shape,
                scale_factor)
        if self.export_onnx:
            fake_bboxes = tensorlayerx.convert_to_tensor(np.array([[0.0, 
                0.0, 0.0, 0.0, 1.0, 1.0]], dtype='float32'))
            bbox_pred = tensorlayerx.concat([bbox_pred, fake_bboxes])
            bbox_num = bbox_num + 1
        return bbox_pred, bbox_num

    def get_pred(self, bboxes, bbox_num, im_shape, scale_factor):
        """
        Rescale, clip and filter the bbox from the output of NMS to 
        get final prediction. 

        Notes:
        Currently only support bs = 1.

        Args:
            bboxes (Tensor): The output bboxes with shape [N, 6] after decode
                and NMS, including labels, scores and bboxes.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
        Returns:
            pred_result (Tensor): The final prediction results with shape [N, 6]
                including labels, scores and bboxes.
        """
        if self.export_eb:
            return bboxes, bboxes, bbox_num
        if not self.export_onnx:
            bboxes_list = []
            bbox_num_list = []
            id_start = 0
            fake_bboxes = tensorlayerx.convert_to_tensor(np.array([[0.0, 
                0.0, 0.0, 0.0, 1.0, 1.0]], dtype='float32'))
            fake_bbox_num = tensorlayerx.convert_to_tensor(np.array([1],
                dtype='int32'))
            for i in range(bbox_num.shape[0]):
                if bbox_num[i] == 0:
                    bboxes_i = fake_bboxes
                    bbox_num_i = fake_bbox_num
                else:
                    bboxes_i = bboxes[id_start:id_start + bbox_num[i], :]
                    bbox_num_i = bbox_num[i]
                    id_start += bbox_num[i]
                bboxes_list.append(bboxes_i)
                bbox_num_list.append(bbox_num_i)
            bboxes = tensorlayerx.concat(bboxes_list)
            bbox_num = tensorlayerx.concat(bbox_num_list)
        origin_shape = tensorlayerx.floor(im_shape / scale_factor + 0.5)
        if not self.export_onnx:
            origin_shape_list = []
            scale_factor_list = []
            for i in range(bbox_num.shape[0]):
                expand_shape = paddle.expand(origin_shape[i:i + 1, :], [
                    bbox_num[i], 2])
                scale_y, scale_x = scale_factor[i][0], scale_factor[i][1]
                scale = tensorlayerx.concat([scale_x, scale_y, scale_x,
                    scale_y])
                expand_scale = paddle.expand(scale, [bbox_num[i], 4])
                origin_shape_list.append(expand_shape)
                scale_factor_list.append(expand_scale)
            self.origin_shape_list = tensorlayerx.concat(origin_shape_list)
            scale_factor_list = tensorlayerx.concat(scale_factor_list)
        else:
            scale_y, scale_x = scale_factor[0][0], scale_factor[0][1]
            scale = tensorlayerx.concat([scale_x, scale_y, scale_x, scale_y]
                ).unsqueeze(0)
            self.origin_shape_list = paddle.expand(origin_shape, [bbox_num[
                0], 2])
            scale_factor_list = paddle.expand(scale, [bbox_num[0], 4])
        pred_label = bboxes[:, 0:1]
        pred_score = bboxes[:, 1:2]
        pred_bbox = bboxes[:, 2:]
        scaled_bbox = pred_bbox / scale_factor_list
        origin_h = self.origin_shape_list[:, 0]
        origin_w = self.origin_shape_list[:, 1]
        zeros = tensorlayerx.zeros_like(origin_h)
        x1 = tensorlayerx.maximum(tensorlayerx.minimum(scaled_bbox[:, 0],
            origin_w), zeros)
        y1 = tensorlayerx.maximum(tensorlayerx.minimum(scaled_bbox[:, 1],
            origin_h), zeros)
        x2 = tensorlayerx.maximum(tensorlayerx.minimum(scaled_bbox[:, 2],
            origin_w), zeros)
        y2 = tensorlayerx.maximum(tensorlayerx.minimum(scaled_bbox[:, 3],
            origin_h), zeros)
        pred_bbox = tensorlayerx.ops.stack([x1, y1, x2, y2], axis=-1)
        keep_mask = nonempty_bbox(pred_bbox, return_mask=True)
        keep_mask = tensorlayerx.expand_dims(keep_mask, [1])
        pred_label = tensorlayerx.where(keep_mask, pred_label, paddle.
            ones_like(pred_label) * -1)
        pred_result = tensorlayerx.concat([pred_label, pred_score,
            pred_bbox], axis=1)
        return bboxes, pred_result, bbox_num

    def get_origin_shape(self):
        return self.origin_shape_list


@register
class MaskPostProcess(object):
    __shared__ = ['export_onnx', 'assign_on_cpu']
    """
    refer to:
    https://github.com/facebookresearch/detectron2/layers/mask_ops.py

    Get Mask output according to the output from model
    """

    def __init__(self, binary_thresh=0.5, export_onnx=False, assign_on_cpu=\
        False):
        super(MaskPostProcess, self).__init__()
        self.binary_thresh = binary_thresh
        self.export_onnx = export_onnx
        self.assign_on_cpu = assign_on_cpu

    def paste_mask(self, masks, boxes, im_h, im_w):
        """
        Paste the mask prediction to the original image.
        """
        x0_int, y0_int = 0, 0
        x1_int, y1_int = im_w, im_h
        x0, y0, x1, y1 = tensorlayerx.ops.split(boxes, 4, axis=1)
        N = masks.shape[0]
        aa = tensorlayerx.ops.arange(y0_int, y1_int)
        bb = tensorlayerx.ops.arange(x0_int, x1_int)
        img_y = aa + 0.5
        img_x = bb + 0.5
        img_y = (img_y - y0) / (y1 - y0) * 2 - 1
        img_x = (img_x - x0) / (x1 - x0) * 2 - 1
        if self.assign_on_cpu:
            tensorlayerx.set_device('cpu')
        gx = img_x[:, None, :].expand([N, paddle2tlx.pd2tlx.ops.tlxops.
            tlx_get_tensor_shape(img_y)[1], paddle2tlx.pd2tlx.ops.tlxops.
            tlx_get_tensor_shape(img_x)[1]])
        gy = img_y[:, :, None].expand([N, paddle2tlx.pd2tlx.ops.tlxops.
            tlx_get_tensor_shape(img_y)[1], paddle2tlx.pd2tlx.ops.tlxops.
            tlx_get_tensor_shape(img_x)[1]])
        grid = tensorlayerx.ops.stack([gx, gy], axis=3)
        img_masks = paddle.nn.functional.grid_sample(masks, grid,
            align_corners=False)
        return img_masks[:, 0]

    def __call__(self, mask_out, bboxes, bbox_num, origin_shape):
        """
        Decode the mask_out and paste the mask to the origin image.

        Args:
            mask_out (Tensor): mask_head output with shape [N, 28, 28].
            bbox_pred (Tensor): The output bboxes with shape [N, 6] after decode
                and NMS, including labels, scores and bboxes.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
            origin_shape (Tensor): The origin shape of the input image, the tensor
                shape is [N, 2], and each row is [h, w].
        Returns:
            pred_result (Tensor): The final prediction mask results with shape
                [N, h, w] in binary mask style.
        """
        num_mask = mask_out.shape[0]
        origin_shape = tensorlayerx.cast(origin_shape, 'int32')
        device = tensorlayerx.get_device()
        if self.export_onnx:
            h, w = origin_shape[0][0], origin_shape[0][1]
            mask_onnx = self.paste_mask(mask_out[:, None, :, :], bboxes[:, 
                2:], h, w)
            mask_onnx = mask_onnx >= self.binary_thresh
            pred_result = tensorlayerx.cast(mask_onnx, 'int32')
        else:
            max_h = tensorlayerx.reduce_max(origin_shape[:, (0)])
            max_w = tensorlayerx.reduce_max(origin_shape[:, (1)])
            aa = tensorlayerx.zeros([num_mask, max_h, max_w], dtype='int32')
            pred_result = aa - 1
            id_start = 0
            for i in range(paddle2tlx.pd2tlx.ops.tlxops.
                tlx_get_tensor_shape(bbox_num)[0]):
                bboxes_i = bboxes[id_start:id_start + bbox_num[i], :]
                mask_out_i = mask_out[id_start:id_start + bbox_num[i], :, :]
                im_h = origin_shape[i, 0]
                im_w = origin_shape[i, 1]
                bbox_num_i = bbox_num[id_start]
                pred_mask = self.paste_mask(mask_out_i[:, None, :, :],
                    bboxes_i[:, 2:], im_h, im_w)
                pred_mask = tensorlayerx.cast(pred_mask >= self.
                    binary_thresh, 'int32')
                pred_result[id_start:id_start + bbox_num[i], :im_h, :im_w
                    ] = pred_mask
                id_start += bbox_num[i]
        if self.assign_on_cpu:
            tensorlayerx.set_device(device)
        return pred_result


@register
class FCOSPostProcess(object):
    __inject__ = ['decode', 'nms']

    def __init__(self, decode=None, nms=None):
        super(FCOSPostProcess, self).__init__()
        self.decode = decode
        self.nms = nms

    def __call__(self, fcos_head_outs, scale_factor):
        """
        Decode the bbox and do NMS in FCOS.
        """
        locations, cls_logits, bboxes_reg, centerness = fcos_head_outs
        bboxes, score = self.decode(locations, cls_logits, bboxes_reg,
            centerness, scale_factor)
        bbox_pred, bbox_num, _ = self.nms(bboxes, score)
        return bbox_pred, bbox_num


@register
class CenterNetPostProcess(TTFBox):
    """
    Postprocess the model outputs to get final prediction:
        1. Do NMS for heatmap to get top `max_per_img` bboxes.
        2. Decode bboxes using center offset and box size.
        3. Rescale decoded bboxes reference to the origin image shape.

    Args:
        max_per_img(int): the maximum number of predicted objects in a image,
            500 by default.
        down_ratio(int): the down ratio from images to heatmap, 4 by default.
        regress_ltrb (bool): whether to regress left/top/right/bottom or
            width/height for a box, true by default.
        for_mot (bool): whether return other features used in tracking model.
    """
    __shared__ = ['down_ratio', 'for_mot']

    def __init__(self, max_per_img=500, down_ratio=4, regress_ltrb=True,
        for_mot=False):
        super(TTFBox, self).__init__()
        self.max_per_img = max_per_img
        self.down_ratio = down_ratio
        self.regress_ltrb = regress_ltrb
        self.for_mot = for_mot

    def __call__(self, hm, wh, reg, im_shape, scale_factor):
        heat = self._simple_nms(hm)
        scores, inds, topk_clses, ys, xs = self._topk(heat)
        scores = scores.unsqueeze(1)
        clses = topk_clses.unsqueeze(1)
        reg_t = tensorlayerx.transpose(reg, [0, 2, 3, 1])
        reg = tensorlayerx.reshape(reg_t, [-1, reg_t.shape[-1]])
        reg = tensorlayerx.gather(reg, inds)
        xs = tensorlayerx.cast(xs, 'float32')
        ys = tensorlayerx.cast(ys, 'float32')
        xs = xs + reg[:, 0:1]
        ys = ys + reg[:, 1:2]
        wh_t = tensorlayerx.transpose(wh, [0, 2, 3, 1])
        wh = tensorlayerx.reshape(wh_t, [-1, wh_t.shape[-1]])
        wh = tensorlayerx.gather(wh, inds)
        if self.regress_ltrb:
            x1 = xs - wh[:, 0:1]
            y1 = ys - wh[:, 1:2]
            x2 = xs + wh[:, 2:3]
            y2 = ys + wh[:, 3:4]
        else:
            x1 = xs - wh[:, 0:1] / 2
            y1 = ys - wh[:, 1:2] / 2
            x2 = xs + wh[:, 0:1] / 2
            y2 = ys + wh[:, 1:2] / 2
        n, c, feat_h, feat_w = (paddle2tlx.pd2tlx.ops.tlxops.
            tlx_get_tensor_shape(hm))
        padw = (feat_w * self.down_ratio - im_shape[0, 1]) / 2
        padh = (feat_h * self.down_ratio - im_shape[0, 0]) / 2
        x1 = x1 * self.down_ratio
        y1 = y1 * self.down_ratio
        x2 = x2 * self.down_ratio
        y2 = y2 * self.down_ratio
        x1 = x1 - padw
        y1 = y1 - padh
        x2 = x2 - padw
        y2 = y2 - padh
        bboxes = tensorlayerx.concat([x1, y1, x2, y2], axis=1)
        scale_y = scale_factor[:, 0:1]
        scale_x = scale_factor[:, 1:2]
        scale_expand = tensorlayerx.concat([scale_x, scale_y, scale_x,
            scale_y], axis=1)
        boxes_shape = bboxes.shape[:]
        scale_expand = paddle.expand(scale_expand, shape=boxes_shape)
        bboxes = tensorlayerx.divide(bboxes, scale_expand)
        results = tensorlayerx.concat([clses, scores, bboxes], axis=1)
        if self.for_mot:
            return results, inds, topk_clses
        else:
            return results, paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(
                results)[0:1], topk_clses


@register
class SparsePostProcess(object):
    __shared__ = ['num_classes']

    def __init__(self, num_proposals, num_classes=80):
        super(SparsePostProcess, self).__init__()
        self.num_classes = num_classes
        self.num_proposals = num_proposals

    def __call__(self, box_cls, box_pred, scale_factor_wh, img_whwh):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            scale_factor_wh (Tensor): tensors of shape [batch_size, 2] the scalor of  per img
            img_whwh (Tensor): tensors of shape [batch_size, 4]
        Returns:
            bbox_pred (Tensor): tensors of shape [num_boxes, 6] Each row has 6 values:
            [label, confidence, xmin, ymin, xmax, ymax]
            bbox_num (Tensor): tensors of shape [batch_size] the number of RoIs in each image.
        """
        assert len(box_cls) == len(scale_factor_wh) == len(img_whwh)
        img_wh = img_whwh[:, :2]
        scores = tensorlayerx.ops.sigmoid(box_cls)
        labels = tensorlayerx.ops.arange(0, self.num_classes).unsqueeze(0
            ).tile([self.num_proposals, 1]).flatten(start_axis=0, stop_axis=1)
        classes_all = []
        scores_all = []
        boxes_all = []
        for i, (scores_per_image, box_pred_per_image) in enumerate(zip(
            scores, box_pred)):
            scores_per_image, topk_indices = scores_per_image.flatten(0, 1
                ).topk(self.num_proposals, sorted=False)
            labels_per_image = tensorlayerx.gather(labels, topk_indices, axis=0
                )
            box_pred_per_image = box_pred_per_image.reshape([-1, 1, 4]).tile([
                1, self.num_classes, 1]).reshape([-1, 4])
            box_pred_per_image = tensorlayerx.gather(box_pred_per_image,
                topk_indices, axis=0)
            classes_all.append(labels_per_image)
            scores_all.append(scores_per_image)
            boxes_all.append(box_pred_per_image)
        bbox_num = tensorlayerx.zeros([len(scale_factor_wh)], dtype='int32')
        boxes_final = []
        for i in range(len(scale_factor_wh)):
            classes = classes_all[i]
            boxes = boxes_all[i]
            scores = scores_all[i]
            aa = tensorlayerx.ops.clip_by_value(boxes[:, 0::2],
                clip_value_min=0, clip_value_max=img_wh[i][0])
            bb = tensorlayerx.ops.clip_by_value(boxes[:, 1::2],
                clip_value_min=0, clip_value_max=img_wh[i][1])
            boxes[:, 0::2] = aa / scale_factor_wh[i][0]
            boxes[:, 1::2] = bb / scale_factor_wh[i][1]
            boxes_w, boxes_h = (boxes[:, (2)] - boxes[:, (0)]).numpy(), (boxes
                [:, (3)] - boxes[:, (1)]).numpy()
            keep = (boxes_w > 1.0) & (boxes_h > 1.0)
            if keep.sum() == 0:
                bboxes = tensorlayerx.zeros([1, 6]).astype('float32')
            else:
                boxes = tensorlayerx.convert_to_tensor(boxes.numpy()[keep]
                    ).astype('float32')
                classes = tensorlayerx.convert_to_tensor(classes.numpy()[keep]
                    ).astype('float32').unsqueeze(-1)
                scores = tensorlayerx.convert_to_tensor(scores.numpy()[keep]
                    ).astype('float32').unsqueeze(-1)
                bboxes = tensorlayerx.concat([classes, scores, boxes], axis=-1)
            boxes_final.append(bboxes)
            bbox_num[i] = bboxes.shape[0]
        bbox_pred = tensorlayerx.concat(boxes_final)
        return bbox_pred, bbox_num


def multiclass_nms(bboxs, num_classes, match_threshold=0.6, match_metric='iou'
    ):
    final_boxes = []
    for c in range(num_classes):
        idxs = bboxs[:, 0] == c
        if np.count_nonzero(idxs) == 0:
            continue
        r = nms(bboxs[idxs, 1:], match_threshold, match_metric)
        final_boxes.append(np.concatenate([np.full((r.shape[0], 1), c), r], 1))
    return final_boxes


def nms(dets, match_threshold=0.6, match_metric='iou'):
    """ Apply NMS to avoid detecting too many overlapping bounding boxes.
        Args:
            dets: shape [N, 5], [score, x1, y1, x2, y2]
            match_metric: 'iou' or 'ios'
            match_threshold: overlap thresh for match metric.
    """
    if dets.shape[0] == 0:
        return dets[[], :]
    scores = dets[:, 0]
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    ndets = dets.shape[0]
    suppressed = np.zeros(ndets, dtype=np.int)
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            if match_metric == 'iou':
                union = iarea + areas[j] - inter
                match_value = inter / union
            elif match_metric == 'ios':
                smaller = min(iarea, areas[j])
                match_value = inter / smaller
            else:
                raise ValueError()
            if match_value >= match_threshold:
                suppressed[j] = 1
    keep = np.where(suppressed == 0)[0]
    dets = dets[keep, :]
    return dets
