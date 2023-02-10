from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
from tensorlayerx.nn.initializers import xavier_uniform
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import random_normal
from tensorlayerx.nn.initializers import Constant
from models.layers import ConvNormLayer
from models.layers import MaskMatrixNMS
from models.layers import DropBlock
from core.workspace import register
from six.moves import zip
import numpy as np
__all__ = ['SOLOv2Head']


@register
class SOLOv2MaskHead(nn.Module):
    """
    MaskHead of SOLOv2.
    The code of this function is based on:
        https://github.com/WXinlong/SOLO/blob/master/mmdet/models/mask_heads/mask_feat_head.py

    Args:
        in_channels (int): The channel number of input Tensor.
        out_channels (int): The channel number of output Tensor.
        start_level (int): The position where the input starts.
        end_level (int): The position where the input ends.
        use_dcn_in_tower (bool): Whether to use dcn in tower or not.
    """
    __shared__ = ['norm_type']

    def __init__(self, in_channels=256, mid_channels=128, out_channels=256,
        start_level=0, end_level=3, use_dcn_in_tower=False, norm_type='bn'):
        super(SOLOv2MaskHead, self).__init__()
        assert start_level >= 0 and end_level >= start_level
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.use_dcn_in_tower = use_dcn_in_tower
        self.range_level = end_level - start_level + 1
        self.use_dcn = True if self.use_dcn_in_tower else False
        self.convs_all_levels = []
        self.norm_type = norm_type
        for i in range(start_level, end_level + 1):
            conv_feat_name = 'mask_feat_head.convs_all_levels.{}'.format(i)
            conv_pre_feat = []
            if i == start_level:
                conv_pre_feat.append(ConvNormLayer(ch_in=self.in_channels,
                    ch_out=self.mid_channels, filter_size=3, stride=1,
                    use_dcn=self.use_dcn, norm_type=self.norm_type))
                self.add_sublayer('conv_pre_feat' + str(i), nn.Sequential([
                    *conv_pre_feat]))
                self.convs_all_levels.append(nn.Sequential([*conv_pre_feat]))
            else:
                for j in range(i):
                    ch_in = 0
                    if j == 0:
                        ch_in = (self.in_channels + 2 if i == end_level else
                            self.in_channels)
                    else:
                        ch_in = self.mid_channels
                    conv_pre_feat.append(ConvNormLayer(ch_in=ch_in, ch_out=\
                        self.mid_channels, filter_size=3, stride=1, use_dcn
                        =self.use_dcn, norm_type=self.norm_type))
                    conv_pre_feat.append(nn.ReLU())
                    conv_pre_feat.append(paddle2tlx.pd2tlx.ops.tlxops.
                        tlx_Upsample(scale_factor=2, mode='bilinear',
                        data_format='channels_first'))
                self.add_sublayer('conv_pre_feat' + str(i), nn.Sequential([
                    *conv_pre_feat]))
                self.convs_all_levels.append(nn.Sequential([*conv_pre_feat]))
        conv_pred_name = 'mask_feat_head.conv_pred.0'
        self.conv_pred = self.add_sublayer(conv_pred_name, ConvNormLayer(
            ch_in=self.mid_channels, ch_out=self.out_channels, filter_size=\
            1, stride=1, use_dcn=self.use_dcn, norm_type=self.norm_type))

    def forward(self, inputs):
        """
        Get SOLOv2MaskHead output.

        Args:
            inputs(list[Tensor]): feature map from each necks with shape of [N, C, H, W]
        Returns:
            ins_pred(Tensor): Output of SOLOv2MaskHead head
        """
        feat_all_level = tensorlayerx.ops.relu(self.convs_all_levels[0](
            inputs[0]))
        for i in range(1, self.range_level):
            input_p = inputs[i]
            if i == self.range_level - 1:
                input_feat = input_p
                x_range = paddle2tlx.pd2tlx.ops.tlxops.tlx_linspace(-1, 1,
                    paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(
                    input_feat)[-1], dtype='float32')
                y_range = paddle2tlx.pd2tlx.ops.tlxops.tlx_linspace(-1, 1,
                    paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(
                    input_feat)[-2], dtype='float32')
                y, x = tensorlayerx.meshgrid([y_range, x_range])
                x = tensorlayerx.expand_dims(x, [0, 1])
                y = tensorlayerx.expand_dims(y, [0, 1])
                y = paddle.expand(y, shape=[paddle2tlx.pd2tlx.ops.tlxops.
                    tlx_get_tensor_shape(input_feat)[0], 1, -1, -1])
                x = paddle.expand(x, shape=[paddle2tlx.pd2tlx.ops.tlxops.
                    tlx_get_tensor_shape(input_feat)[0], 1, -1, -1])
                coord_feat = tensorlayerx.concat([x, y], axis=1)
                input_p = tensorlayerx.concat([input_p, coord_feat], axis=1)
            feat_all_level = tensorlayerx.add(feat_all_level, self.
                convs_all_levels[i](input_p))
        ins_pred = tensorlayerx.ops.relu(self.conv_pred(feat_all_level))
        return ins_pred


@register
class SOLOv2Head(nn.Module):
    """
    Head block for SOLOv2 network

    Args:
        num_classes (int): Number of output classes.
        in_channels (int): Number of input channels.
        seg_feat_channels (int): Num_filters of kernel & categroy branch convolution operation.
        stacked_convs (int): Times of convolution operation.
        num_grids (list[int]): List of feature map grids size.
        kernel_out_channels (int): Number of output channels in kernel branch.
        dcn_v2_stages (list): Which stage use dcn v2 in tower. It is between [0, stacked_convs).
        segm_strides (list[int]): List of segmentation area stride.
        solov2_loss (object): SOLOv2Loss instance.
        score_threshold (float): Threshold of categroy score.
        mask_nms (object): MaskMatrixNMS instance.
    """
    __inject__ = ['solov2_loss', 'mask_nms']
    __shared__ = ['norm_type', 'num_classes']

    def __init__(self, num_classes=80, in_channels=256, seg_feat_channels=\
        256, stacked_convs=4, num_grids=[40, 36, 24, 16, 12],
        kernel_out_channels=256, dcn_v2_stages=[], segm_strides=[8, 8, 16, 
        32, 32], solov2_loss=None, score_threshold=0.1, mask_threshold=0.5,
        mask_nms=None, norm_type='bn', drop_block=False):
        super(SOLOv2Head, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = kernel_out_channels
        self.dcn_v2_stages = dcn_v2_stages
        self.segm_strides = segm_strides
        self.solov2_loss = solov2_loss
        self.mask_nms = mask_nms
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.norm_type = norm_type
        self.drop_block = drop_block
        self.kernel_pred_convs = []
        self.cate_pred_convs = []
        for i in range(self.stacked_convs):
            use_dcn = True if i in self.dcn_v2_stages else False
            ch_in = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            kernel_conv = self.add_sublayer('bbox_head.kernel_convs.' + str
                (i), ConvNormLayer(ch_in=ch_in, ch_out=self.
                seg_feat_channels, filter_size=3, stride=1, use_dcn=use_dcn,
                norm_type=self.norm_type))
            self.kernel_pred_convs.append(kernel_conv)
            ch_in = self.in_channels if i == 0 else self.seg_feat_channels
            cate_conv = self.add_sublayer('bbox_head.cate_convs.' + str(i),
                ConvNormLayer(ch_in=ch_in, ch_out=self.seg_feat_channels,
                filter_size=3, stride=1, use_dcn=use_dcn, norm_type=self.
                norm_type))
            self.cate_pred_convs.append(cate_conv)
        self.solo_kernel = self.add_sublayer('bbox_head.solo_kernel', nn.
            GroupConv2d(kernel_size=3, stride=1, padding=1, in_channels=\
            self.seg_feat_channels, out_channels=self.kernel_out_channels,
            W_init=xavier_uniform(), data_format='channels_first'))
        self.solo_cate = self.add_sublayer('bbox_head.solo_cate', nn.
            GroupConv2d(kernel_size=3, stride=1, padding=1, in_channels=\
            self.seg_feat_channels, out_channels=self.cate_out_channels,
            W_init=xavier_uniform(), b_init=xavier_uniform(), data_format=\
            'channels_first'))
        if self.drop_block and self.training:
            self.drop_block_fun = DropBlock(block_size=3, keep_prob=0.9,
                name='solo_cate.dropblock')

    def _points_nms(self, heat, kernel_size=2):
        hmax = paddle.nn.functional.max_pool2d(heat, kernel_size=\
            kernel_size, stride=1, padding=1)
        keep = tensorlayerx.cast(hmax[:, :, :-1, :-1] == heat, 'float32')
        return heat * keep

    def _split_feats(self, feats):
        return paddle.nn.functional.interpolate(feats[0], scale_factor=0.5,
            align_corners=False, align_mode=0, mode='bilinear'), feats[1
            ], feats[2], feats[3], paddle.nn.functional.interpolate(feats[4
            ], size=paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(feats
            [3])[-2:], mode='bilinear', align_corners=False, align_mode=0)

    def forward(self, input):
        """
        Get SOLOv2 head output

        Args:
            input (list): List of Tensors, output of backbone or neck stages
        Returns:
            cate_pred_list (list): Tensors of each category branch layer
            kernel_pred_list (list): Tensors of each kernel branch layer
        """
        feats = self._split_feats(input)
        cate_pred_list = []
        kernel_pred_list = []
        for idx in range(len(self.seg_num_grids)):
            cate_pred, kernel_pred = self._get_output_single(feats[idx], idx)
            cate_pred_list.append(cate_pred)
            kernel_pred_list.append(kernel_pred)
        return cate_pred_list, kernel_pred_list

    def _get_output_single(self, input, idx):
        ins_kernel_feat = input
        x_range = paddle2tlx.pd2tlx.ops.tlxops.tlx_linspace(-1, 1,
            paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(
            ins_kernel_feat)[-1], dtype='float32')
        y_range = paddle2tlx.pd2tlx.ops.tlxops.tlx_linspace(-1, 1,
            paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(
            ins_kernel_feat)[-2], dtype='float32')
        y, x = tensorlayerx.meshgrid([y_range, x_range])
        x = tensorlayerx.expand_dims(x, [0, 1])
        y = tensorlayerx.expand_dims(y, [0, 1])
        y = paddle.expand(y, shape=[paddle2tlx.pd2tlx.ops.tlxops.
            tlx_get_tensor_shape(ins_kernel_feat)[0], 1, -1, -1])
        x = paddle.expand(x, shape=[paddle2tlx.pd2tlx.ops.tlxops.
            tlx_get_tensor_shape(ins_kernel_feat)[0], 1, -1, -1])
        coord_feat = tensorlayerx.concat([x, y], axis=1)
        ins_kernel_feat = tensorlayerx.concat([ins_kernel_feat, coord_feat],
            axis=1)
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[idx]
        kernel_feat = paddle.nn.functional.interpolate(kernel_feat, size=[
            seg_num_grid, seg_num_grid], mode='bilinear', align_corners=\
            False, align_mode=0)
        cate_feat = kernel_feat[:, :-2, :, :]
        for kernel_layer in self.kernel_pred_convs:
            kernel_feat = tensorlayerx.ops.relu(kernel_layer(kernel_feat))
        if self.drop_block and self.training:
            kernel_feat = self.drop_block_fun(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)
        for cate_layer in self.cate_pred_convs:
            cate_feat = tensorlayerx.ops.relu(cate_layer(cate_feat))
        if self.drop_block and self.training:
            cate_feat = self.drop_block_fun(cate_feat)
        cate_pred = self.solo_cate(cate_feat)
        if not self.training:
            cate_pred = self._points_nms(tensorlayerx.ops.sigmoid(cate_pred
                ), kernel_size=2)
            cate_pred = tensorlayerx.transpose(cate_pred, [0, 2, 3, 1])
        return cate_pred, kernel_pred

    def get_loss(self, cate_preds, kernel_preds, ins_pred, ins_labels,
        cate_labels, grid_order_list, fg_num):
        """
        Get loss of network of SOLOv2.

        Args:
            cate_preds (list): Tensor list of categroy branch output.
            kernel_preds (list): Tensor list of kernel branch output.
            ins_pred (list): Tensor list of instance branch output.
            ins_labels (list): List of instance labels pre batch.
            cate_labels (list): List of categroy labels pre batch.
            grid_order_list (list): List of index in pre grid.
            fg_num (int): Number of positive samples in a mini-batch.
        Returns:
            loss_ins (Tensor): The instance loss Tensor of SOLOv2 network.
            loss_cate (Tensor): The category loss Tensor of SOLOv2 network.
        """
        batch_size = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(
            grid_order_list[0])[0]
        ins_pred_list = []
        for kernel_preds_level, grid_orders_level in zip(kernel_preds,
            grid_order_list):
            if grid_orders_level.shape[1] == 0:
                ins_pred_list.append(None)
                continue
            grid_orders_level = tensorlayerx.reshape(grid_orders_level, [-1])
            reshape_pred = tensorlayerx.reshape(kernel_preds_level, shape=(
                paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(
                kernel_preds_level)[0], paddle2tlx.pd2tlx.ops.tlxops.
                tlx_get_tensor_shape(kernel_preds_level)[1], -1))
            reshape_pred = tensorlayerx.transpose(reshape_pred, [0, 2, 1])
            reshape_pred = tensorlayerx.reshape(reshape_pred, shape=(-1,
                paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(
                reshape_pred)[2]))
            gathered_pred = tensorlayerx.gather(reshape_pred, indices=\
                grid_orders_level)
            gathered_pred = tensorlayerx.reshape(gathered_pred, shape=[
                batch_size, -1, paddle2tlx.pd2tlx.ops.tlxops.
                tlx_get_tensor_shape(gathered_pred)[1]])
            cur_ins_pred = ins_pred
            cur_ins_pred = tensorlayerx.reshape(cur_ins_pred, shape=(
                paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(
                cur_ins_pred)[0], paddle2tlx.pd2tlx.ops.tlxops.
                tlx_get_tensor_shape(cur_ins_pred)[1], -1))
            ins_pred_conv = tensorlayerx.ops.matmul(gathered_pred, cur_ins_pred
                )
            cur_ins_pred = tensorlayerx.reshape(ins_pred_conv, shape=(-1,
                paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(ins_pred)
                [-2], paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(
                ins_pred)[-1]))
            ins_pred_list.append(cur_ins_pred)
        num_ins = tensorlayerx.reduce_sum(fg_num)
        cate_preds = [tensorlayerx.reshape(tensorlayerx.transpose(cate_pred,
            [0, 2, 3, 1]), shape=(-1, self.cate_out_channels)) for
            cate_pred in cate_preds]
        flatten_cate_preds = tensorlayerx.concat(cate_preds)
        new_cate_labels = []
        for cate_label in cate_labels:
            new_cate_labels.append(tensorlayerx.reshape(cate_label, shape=[-1])
                )
        cate_labels = tensorlayerx.concat(new_cate_labels)
        loss_ins, loss_cate = self.solov2_loss(ins_pred_list, ins_labels,
            flatten_cate_preds, cate_labels, num_ins)
        return {'loss_ins': loss_ins, 'loss_cate': loss_cate}

    def get_prediction(self, cate_preds, kernel_preds, seg_pred, im_shape,
        scale_factor):
        """
        Get prediction result of SOLOv2 network

        Args:
            cate_preds (list): List of Variables, output of categroy branch.
            kernel_preds (list): List of Variables, output of kernel branch.
            seg_pred (list): List of Variables, output of mask head stages.
            im_shape (Variables): [h, w] for input images.
            scale_factor (Variables): [scale, scale] for input images.
        Returns:
            seg_masks (Tensor): The prediction segmentation.
            cate_labels (Tensor): The prediction categroy label of each segmentation.
            seg_masks (Tensor): The prediction score of each segmentation.
        """
        num_levels = len(cate_preds)
        featmap_size = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(
            seg_pred)[-2:]
        seg_masks_list = []
        cate_labels_list = []
        cate_scores_list = []
        cate_preds = [(cate_pred * 1.0) for cate_pred in cate_preds]
        kernel_preds = [(kernel_pred * 1.0) for kernel_pred in kernel_preds]
        for idx in range(1):
            cate_pred_list = [tensorlayerx.reshape(cate_preds[i][idx],
                shape=(-1, self.cate_out_channels)) for i in range(num_levels)]
            seg_pred_list = seg_pred
            kernel_pred_list = [tensorlayerx.reshape(tensorlayerx.transpose
                (kernel_preds[i][idx], [1, 2, 0]), shape=(-1, self.
                kernel_out_channels)) for i in range(num_levels)]
            cate_pred_list = tensorlayerx.concat(cate_pred_list, axis=0)
            kernel_pred_list = tensorlayerx.concat(kernel_pred_list, axis=0)
            seg_masks, cate_labels, cate_scores = self.get_seg_single(
                cate_pred_list, seg_pred_list, kernel_pred_list,
                featmap_size, im_shape[idx], scale_factor[idx][0])
            bbox_num = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(
                cate_labels)[0]
        return seg_masks, cate_labels, cate_scores, bbox_num

    def get_seg_single(self, cate_preds, seg_preds, kernel_preds,
        featmap_size, im_shape, scale_factor):
        """
        The code of this function is based on:
            https://github.com/WXinlong/SOLO/blob/master/mmdet/models/anchor_heads/solov2_head.py#L385
        """
        h = tensorlayerx.cast(im_shape[0], 'int32')[0]
        w = tensorlayerx.cast(im_shape[1], 'int32')[0]
        upsampled_size_out = [featmap_size[0] * 4, featmap_size[1] * 4]
        y = tensorlayerx.zeros(shape=paddle2tlx.pd2tlx.ops.tlxops.
            tlx_get_tensor_shape(cate_preds), dtype='float32')
        inds = tensorlayerx.where(cate_preds > self.score_threshold,
            cate_preds, y)
        inds = paddle2tlx.pd2tlx.ops.tlxops.tlx_nonzero(inds)
        cate_preds = tensorlayerx.reshape(cate_preds, shape=[-1])
        ind_a = tensorlayerx.convert_to_tensor(paddle2tlx.pd2tlx.ops.tlxops
            .tlx_get_tensor_shape(kernel_preds)[0], 'int64')
        ind_b = tensorlayerx.zeros(shape=[1], dtype='int64')
        inds_end = tensorlayerx.expand_dims(tensorlayerx.concat([ind_a,
            ind_b]), 0)
        inds = tensorlayerx.concat([inds, inds_end])
        kernel_preds_end = tensorlayerx.ones(shape=[1, self.
            kernel_out_channels], dtype='float32')
        kernel_preds = tensorlayerx.concat([kernel_preds, kernel_preds_end])
        cate_preds = tensorlayerx.concat([cate_preds, tensorlayerx.zeros(
            shape=[1], dtype='float32')])
        cate_labels = inds[:, 1]
        kernel_preds = tensorlayerx.gather(kernel_preds, indices=inds[:, (0)])
        cate_score_idx = tensorlayerx.add(inds[:, (0)] * self.
            cate_out_channels, cate_labels)
        cate_scores = tensorlayerx.gather(cate_preds, indices=cate_score_idx)
        size_trans = np.power(self.seg_num_grids, 2)
        strides = []
        for _ind in range(len(self.segm_strides)):
            strides.append(tensorlayerx.constant(shape=[int(size_trans[_ind
                ])], dtype='int32', value=self.segm_strides[_ind]))
        strides = tensorlayerx.concat(strides)
        strides = tensorlayerx.concat([strides, tensorlayerx.zeros(shape=[1
            ], dtype='int32')])
        strides = tensorlayerx.gather(strides, indices=inds[:, (0)])
        kernel_preds = tensorlayerx.expand_dims(kernel_preds, [2, 3])
        seg_preds = paddle.nn.functional.conv2d(seg_preds, kernel_preds)
        seg_preds = tensorlayerx.ops.sigmoid(tensorlayerx.ops.squeeze(
            seg_preds, [0]))
        seg_masks = seg_preds > self.mask_threshold
        seg_masks = tensorlayerx.cast(seg_masks, 'float32')
        sum_masks = tensorlayerx.reduce_sum(seg_masks, axis=[1, 2])
        y = tensorlayerx.zeros(shape=paddle2tlx.pd2tlx.ops.tlxops.
            tlx_get_tensor_shape(sum_masks), dtype='float32')
        keep = tensorlayerx.where(sum_masks > strides, sum_masks, y)
        keep = paddle2tlx.pd2tlx.ops.tlxops.tlx_nonzero(keep)
        keep = tensorlayerx.ops.squeeze(keep, axis=[1])
        aa = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(sum_masks)
        keep_other = tensorlayerx.concat([keep, tensorlayerx.
            convert_to_tensor(aa[0] - 1, 'int64')])
        keep_scores = tensorlayerx.concat([keep, tensorlayerx.
            convert_to_tensor(paddle2tlx.pd2tlx.ops.tlxops.
            tlx_get_tensor_shape(sum_masks)[0], 'int64')])
        cate_scores_end = tensorlayerx.zeros(shape=[1], dtype='float32')
        cate_scores = tensorlayerx.concat([cate_scores, cate_scores_end])
        seg_masks = tensorlayerx.gather(seg_masks, indices=keep_other)
        seg_preds = tensorlayerx.gather(seg_preds, indices=keep_other)
        sum_masks = tensorlayerx.gather(sum_masks, indices=keep_other)
        cate_labels = tensorlayerx.gather(cate_labels, indices=keep_other)
        cate_scores = tensorlayerx.gather(cate_scores, indices=keep_scores)
        seg_mul = tensorlayerx.cast(seg_preds * seg_masks, 'float32')
        aa = tensorlayerx.reduce_sum(seg_mul, axis=[1, 2])
        seg_scores = aa / sum_masks
        cate_scores *= seg_scores
        seg_preds, cate_scores, cate_labels = self.mask_nms(seg_preds,
            seg_masks, cate_labels, cate_scores, sum_masks=sum_masks)
        ori_shape = im_shape[:2] / scale_factor + 0.5
        ori_shape = tensorlayerx.cast(ori_shape, 'int32')
        seg_preds = paddle.nn.functional.interpolate(tensorlayerx.
            expand_dims(seg_preds, 0), size=upsampled_size_out, mode=\
            'bilinear', align_corners=False, align_mode=0)
        seg_preds = paddle.slice(seg_preds, axes=[2, 3], starts=[0, 0],
            ends=[h, w])
        seg_masks = tensorlayerx.ops.squeeze(paddle.nn.functional.
            interpolate(seg_preds, size=ori_shape[:2], mode='bilinear',
            align_corners=False, align_mode=0), axis=[0])
        seg_masks = tensorlayerx.cast(seg_masks > self.mask_threshold, 'uint8')
        return seg_masks, cate_labels, cate_scores
