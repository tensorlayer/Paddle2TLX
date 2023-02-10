import tensorlayerx as tlx
import paddle
import paddle2tlx
import six
import numpy as np


def get_det_res(bboxes, bbox_nums, image_id, label_to_cat_id_map, bias=0):
    det_res = []
    k = 0
    for i in range(len(bbox_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = bbox_nums[i]
        for j in range(det_nums):
            dt = bboxes[k]
            k = k + 1
            num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
            if int(num_id) < 0:
                continue
            category_id = label_to_cat_id_map[int(num_id)]
            w = xmax - xmin + bias
            h = ymax - ymin + bias
            bbox = [xmin, ymin, w, h]
            dt_res = {'image_id': cur_image_id, 'category_id': category_id,
                'bbox': bbox, 'score': score}
            det_res.append(dt_res)
    return det_res


def get_det_poly_res(bboxes, bbox_nums, image_id, label_to_cat_id_map, bias=0):
    det_res = []
    k = 0
    for i in range(len(bbox_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = bbox_nums[i]
        for j in range(det_nums):
            dt = bboxes[k]
            k = k + 1
            num_id, score, x1, y1, x2, y2, x3, y3, x4, y4 = dt.tolist()
            if int(num_id) < 0:
                continue
            category_id = label_to_cat_id_map[int(num_id)]
            rbox = [x1, y1, x2, y2, x3, y3, x4, y4]
            dt_res = {'image_id': cur_image_id, 'category_id': category_id,
                'bbox': rbox, 'score': score}
            det_res.append(dt_res)
    return det_res


def strip_mask(mask):
    row = mask[0, 0, :]
    col = mask[0, :, 0]
    im_h = len(col) - np.count_nonzero(col == -1)
    im_w = len(row) - np.count_nonzero(row == -1)
    return mask[:, :im_h, :im_w]


def get_seg_res(masks, bboxes, mask_nums, image_id, label_to_cat_id_map):
    import pycocotools.mask as mask_util
    seg_res = []
    k = 0
    for i in range(len(mask_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = mask_nums[i]
        mask_i = masks[k:k + det_nums]
        mask_i = strip_mask(mask_i)
        for j in range(det_nums):
            mask = mask_i[j].astype(np.uint8)
            score = float(bboxes[k][1])
            label = int(bboxes[k][0])
            k = k + 1
            if label == -1:
                continue
            cat_id = label_to_cat_id_map[label]
            rle = mask_util.encode(np.array(mask[:, :, None], order='F',
                dtype='uint8'))[0]
            if six.PY3:
                if 'counts' in rle:
                    rle['counts'] = rle['counts'].decode('utf8')
            sg_res = {'image_id': cur_image_id, 'category_id': cat_id,
                'segmentation': rle, 'score': score}
            seg_res.append(sg_res)
    return seg_res
