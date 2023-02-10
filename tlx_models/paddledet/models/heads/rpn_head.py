import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import random_normal
from core.workspace import register
from ..proposal_generator.anchor_generator import AnchorGenerator
from ..proposal_generator.target_layer import RPNTargetAssign
from ..proposal_generator.proposal_generator import ProposalGenerator
from ..cls_utils import _get_class_default_kwargs


class RPNFeat(nn.Module):
    """
    Feature extraction in RPN head

    Args:
        in_channel (int): Input channel
        out_channel (int): Output channel
    """

    def __init__(self, in_channel=1024, out_channel=1024):
        super(RPNFeat, self).__init__()
        self.rpn_conv = nn.GroupConv2d(in_channels=in_channel, out_channels
            =out_channel, kernel_size=3, padding=1, W_init=tensorlayerx.nn.
            initializers.xavier_uniform(), data_format='channels_first')
        self.rpn_conv.skip_quant = True

    def forward(self, feats):
        rpn_feats = []
        for feat in feats:
            rpn_feats.append(tensorlayerx.ops.relu(self.rpn_conv(feat)))
        return rpn_feats


@register
class RPNHead(nn.Module):
    """
    Region Proposal Network

    Args:
        anchor_generator (dict): configure of anchor generation
        rpn_target_assign (dict): configure of rpn targets assignment
        train_proposal (dict): configure of proposals generation
            at the stage of training
        test_proposal (dict): configure of proposals generation
            at the stage of prediction
        in_channel (int): channel of input feature maps which can be
            derived by from_config
    """
    __shared__ = ['export_onnx']
    __inject__ = ['loss_rpn_bbox']

    def __init__(self, anchor_generator=_get_class_default_kwargs(
        AnchorGenerator), rpn_target_assign=_get_class_default_kwargs(
        RPNTargetAssign), train_proposal=_get_class_default_kwargs(
        ProposalGenerator, 12000, 2000), test_proposal=\
        _get_class_default_kwargs(ProposalGenerator), in_channel=1024,
        export_onnx=False, loss_rpn_bbox=None):
        super(RPNHead, self).__init__()
        self.anchor_generator = anchor_generator
        self.rpn_target_assign = rpn_target_assign
        self.train_proposal = train_proposal
        self.test_proposal = test_proposal
        self.export_onnx = export_onnx
        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGenerator(**anchor_generator)
        if isinstance(rpn_target_assign, dict):
            self.rpn_target_assign = RPNTargetAssign(**rpn_target_assign)
        if isinstance(train_proposal, dict):
            self.train_proposal = ProposalGenerator(**train_proposal)
        if isinstance(test_proposal, dict):
            self.test_proposal = ProposalGenerator(**test_proposal)
        self.loss_rpn_bbox = loss_rpn_bbox
        num_anchors = self.anchor_generator.num_anchors
        self.rpn_feat = RPNFeat(in_channel, in_channel)
        self.rpn_rois_score = nn.GroupConv2d(in_channels=in_channel,
            out_channels=num_anchors, kernel_size=1, padding=0, W_init=\
            tensorlayerx.nn.initializers.xavier_uniform(), data_format=\
            'channels_first')
        self.rpn_rois_score.skip_quant = True
        self.rpn_rois_delta = nn.GroupConv2d(in_channels=in_channel,
            out_channels=4 * num_anchors, kernel_size=1, padding=0, W_init=\
            tensorlayerx.nn.initializers.xavier_uniform(), data_format=\
            'channels_first')
        self.rpn_rois_delta.skip_quant = True

    @classmethod
    def from_config(cls, cfg, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channel': input_shape.channels}

    def forward(self, feats, inputs):
        rpn_feats = self.rpn_feat(feats)
        scores = []
        deltas = []
        for rpn_feat in rpn_feats:
            rrs = self.rpn_rois_score(rpn_feat)
            rrd = self.rpn_rois_delta(rpn_feat)
            scores.append(rrs)
            deltas.append(rrd)
        anchors = self.anchor_generator(rpn_feats)
        rois, rois_num = self._gen_proposal(scores, deltas, anchors, inputs)
        if self.training:
            loss = self.get_loss(scores, deltas, anchors, inputs)
            return rois, rois_num, loss
        else:
            return rois, rois_num, None

    def _gen_proposal(self, scores, bbox_deltas, anchors, inputs):
        """
        scores (list[Tensor]): Multi-level scores prediction
        bbox_deltas (list[Tensor]): Multi-level deltas prediction
        anchors (list[Tensor]): Multi-level anchors
        inputs (dict): ground truth info
        """
        prop_gen = self.train_proposal if self.training else self.test_proposal
        im_shape = inputs['im_shape']
        if self.export_onnx:
            onnx_rpn_rois_list = []
            onnx_rpn_prob_list = []
            onnx_rpn_rois_num_list = []
            for rpn_score, rpn_delta, anchor in zip(scores, bbox_deltas,
                anchors):
                (onnx_rpn_rois, onnx_rpn_rois_prob, onnx_rpn_rois_num,
                    onnx_post_nms_top_n) = (prop_gen(scores=rpn_score[0:1],
                    bbox_deltas=rpn_delta[0:1], anchors=anchor, im_shape=\
                    im_shape[0:1]))
                onnx_rpn_rois_list.append(onnx_rpn_rois)
                onnx_rpn_prob_list.append(onnx_rpn_rois_prob)
                onnx_rpn_rois_num_list.append(onnx_rpn_rois_num)
            onnx_rpn_rois = tensorlayerx.concat(onnx_rpn_rois_list)
            onnx_rpn_prob = tensorlayerx.concat(onnx_rpn_prob_list).flatten()
            onnx_top_n = tensorlayerx.convert_to_tensor(onnx_post_nms_top_n
                ).cast('int32')
            onnx_num_rois = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(
                onnx_rpn_prob)[0].cast('int32')
            k = tensorlayerx.minimum(onnx_top_n, onnx_num_rois)
            onnx_topk_prob, onnx_topk_inds = tensorlayerx.ops.topk(
                onnx_rpn_prob, k)
            onnx_topk_rois = tensorlayerx.gather(onnx_rpn_rois, onnx_topk_inds)
        else:
            bs_rois_collect = []
            bs_rois_num_collect = []
            shape_im_shape = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(
                im_shape)
            try:
                batch_size = paddle.slice(shape_im_shape, [0], [0], [1])
            except:
                batch_size = paddle.slice(tensorlayerx.convert_to_tensor(
                    shape_im_shape), [0], [0], [1])
            for i in range(batch_size):
                rpn_rois_list = []
                rpn_prob_list = []
                rpn_rois_num_list = []
                for rpn_score, rpn_delta, anchor in zip(scores, bbox_deltas,
                    anchors):
                    (rpn_rois, rpn_rois_prob, rpn_rois_num, post_nms_top_n) = (
                        prop_gen(scores=rpn_score[i:i + 1], bbox_deltas=\
                        rpn_delta[i:i + 1], anchors=anchor, im_shape=\
                        im_shape[i:i + 1]))
                    rpn_rois_list.append(rpn_rois)
                    rpn_prob_list.append(rpn_rois_prob)
                    rpn_rois_num_list.append(rpn_rois_num)
                if len(scores) > 1:
                    rpn_rois = tensorlayerx.concat(rpn_rois_list)
                    rpn_prob = tensorlayerx.concat(rpn_prob_list).flatten()
                    num_rois = (paddle2tlx.pd2tlx.ops.tlxops.
                        tlx_get_tensor_shape(rpn_prob)[0].cast('int32'))
                    if num_rois > post_nms_top_n:
                        topk_prob, topk_inds = tensorlayerx.ops.topk(rpn_prob,
                            post_nms_top_n)
                        topk_rois = tensorlayerx.gather(rpn_rois, topk_inds)
                    else:
                        topk_rois = rpn_rois
                        topk_prob = rpn_prob
                else:
                    topk_rois = rpn_rois_list[0]
                    topk_prob = rpn_prob_list[0].flatten()
                bs_rois_collect.append(topk_rois)
                bs_rois_num_collect.append(paddle2tlx.pd2tlx.ops.tlxops.
                    tlx_get_tensor_shape(topk_rois)[0])
            bs_rois_num_collect = tensorlayerx.concat(bs_rois_num_collect)
        if self.export_onnx:
            output_rois = [onnx_topk_rois]
            output_rois_num = (paddle2tlx.pd2tlx.ops.tlxops.
                tlx_get_tensor_shape(onnx_topk_rois)[0])
        else:
            output_rois = bs_rois_collect
            output_rois_num = bs_rois_num_collect
        return output_rois, output_rois_num

    def get_loss(self, pred_scores, pred_deltas, anchors, inputs):
        """
        pred_scores (list[Tensor]): Multi-level scores prediction
        pred_deltas (list[Tensor]): Multi-level deltas prediction
        anchors (list[Tensor]): Multi-level anchors
        inputs (dict): ground truth info, including im, gt_bbox, gt_score
        """
        anchors = [tensorlayerx.reshape(a, shape=(-1, 4)) for a in anchors]
        anchors = tensorlayerx.concat(anchors)
        scores = [tensorlayerx.reshape(tensorlayerx.transpose(v, perm=[0, 2,
            3, 1]), shape=(v.shape[0], -1, 1)) for v in pred_scores]
        scores = tensorlayerx.concat(scores, axis=1)
        deltas = [tensorlayerx.reshape(tensorlayerx.transpose(v, perm=[0, 2,
            3, 1]), shape=(v.shape[0], -1, 4)) for v in pred_deltas]
        deltas = tensorlayerx.concat(deltas, axis=1)
        score_tgt, bbox_tgt, loc_tgt, norm = self.rpn_target_assign(inputs,
            anchors)
        scores = tensorlayerx.reshape(scores, shape=-1)
        deltas = tensorlayerx.reshape(deltas, shape=(-1, 4))
        score_tgt = tensorlayerx.concat(score_tgt)
        score_tgt.stop_gradient = True
        pos_mask = score_tgt == 1
        pos_ind = paddle2tlx.pd2tlx.ops.tlxops.tlx_nonzero(pos_mask)
        valid_mask = score_tgt >= 0
        valid_ind = paddle2tlx.pd2tlx.ops.tlxops.tlx_nonzero(valid_mask)
        if valid_ind.shape[0] == 0:
            loss_rpn_cls = tensorlayerx.zeros([1], dtype='float32')
        else:
            score_pred = tensorlayerx.gather(scores, valid_ind)
            score_label = tensorlayerx.gather(score_tgt, valid_ind).cast(
                'float32')
            score_label.stop_gradient = True
            loss_rpn_cls = tensorlayerx.losses.sigmoid_cross_entropy(score_pred
                , score_label, reduction='sum')
        if pos_ind.shape[0] == 0:
            loss_rpn_reg = tensorlayerx.zeros([1], dtype='float32')
        else:
            loc_pred = tensorlayerx.gather(deltas, pos_ind)
            loc_tgt = tensorlayerx.concat(loc_tgt)
            loc_tgt = tensorlayerx.gather(loc_tgt, pos_ind)
            loc_tgt.stop_gradient = True
            if self.loss_rpn_bbox is None:
                loss_rpn_reg = tensorlayerx.ops.abs(loc_pred - loc_tgt).sum()
            else:
                loss_rpn_reg = self.loss_rpn_bbox(loc_pred, loc_tgt).sum()
        return {'loss_rpn_cls': loss_rpn_cls / norm, 'loss_rpn_reg': 
            loss_rpn_reg / norm}
