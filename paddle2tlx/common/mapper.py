# coding: utf-8
from paddle2tlx.api_mapper import *


NN_MAPPER = {
    # basic
    "paddle.nn": ["tensorlayerx.nn", None],
    "paddle.nn.Layer": ["tensorlayerx.nn.Module", None],
    "paddle.nn.LayerList": ["tensorlayerx.nn.ModuleList", None],
    "paddle.nn.ParameterList": ["tensorlayerx.nn.ParameterList", None],
    "paddle.nn.Sequential": ["tensorlayerx.nn.Sequential", SequentialModuleMapper],
    # "paddle.create_parameter": ["tensorlayerx.nn.Parameter", CreateParameterMapper],
    "paddle.nn.Layer.__init__": ["tensorlayerx.nn.Module.__init__", None],  # det

    # nn_net
    "paddle.nn.AdaptiveAvgPool1D": ["tensorlayerx.nn.AdaptiveAvgPool1d", AdaptiveAvgPoolMapper],
    "paddle.nn.AdaptiveAvgPool2D": ["tensorlayerx.nn.AdaptiveAvgPool2d", AdaptiveAvgPoolMapper],
    "paddle.nn.AdaptiveAvgPool3D": ["tensorlayerx.nn.AdaptiveAvgPool3d", AdaptiveAvgPoolMapper],
    "paddle.nn.AdaptiveMaxPool2D": ["tensorlayerx.nn.AdaptiveMaxPool2d", AdaptiveMaxPoolMapper],
    "paddle.nn.AvgPool1D": ["tensorlayerx.nn.AvgPool1d", AvgPoolFuncMapper],
    # "paddle.nn.AvgPool2D": ["tensorlayerx.nn.AvgPool2d", AvgPoolFuncMapper],
    "paddle.nn.AvgPool3D": ["tensorlayerx.nn.AvgPool3d", AvgPoolFuncMapper],
    "paddle.nn.BatchNorm": ["tensorlayerx.nn.BatchNorm", BatchNormModuleMapper],
    "paddle.nn.BatchNorm1D": ["tensorlayerx.nn.BatchNorm1d", BatchNormModuleMapper],
    "paddle.nn.BatchNorm2D": ["tensorlayerx.nn.BatchNorm2d", BatchNormModuleMapper],
    "paddle.nn.BatchNorm3D": ["tensorlayerx.nn.BatchNorm3d", BatchNormModuleMapper],
    # "paddle.nn.Conv1D": ["tensorlayerx.nn.Conv1d", ConvModuleMapper],
    # "paddle.nn.Conv2D": ["tensorlayerx.nn.Conv2d", ConvModuleMapper],
    # "paddle.nn.Conv3D": ["tensorlayerx.nn.Conv3d", ConvModuleMapper],
    "paddle.nn.Conv1D": ["tensorlayerx.nn.GroupConv1d", GroupConvModuleMapper],  # mobilenet
    "paddle.nn.Conv2D": ["tensorlayerx.nn.GroupConv2d", GroupConvModuleMapper],
    "paddle.nn.Conv3D": ["tensorlayerx.nn.GroupConv3d", GroupConvModuleMapper],
    # "paddle.nn.Conv2DTranspose": ["tensorlayerx.nn.ConvTranspose2d", None],
    # "paddle.nn.Dropout": ["tensorlayerx.nn.Dropout", DropoutModuleMapper],
    # "paddle.nn.Dropout2D": ["tensorlayerx.nn.Dropout", DropoutModuleMapper],
    "paddle.nn.Embedding": ["tensorlayerx.nn.Embedding", None],
    "paddle.nn.LayerNorm": ["tensorlayerx.nn.LayerNorm", LayerNormModuleMapper],
    "paddle.nn.Linear": ["tensorlayerx.nn.Linear", LinearModuleMapper],
    "paddle.nn.MaxPool1D": ["tensorlayerx.nn.MaxPool1d", MaxPoolModuleMapper],
    # "paddle.nn.MaxPool2D": ["tensorlayerx.nn.MaxPool2d", MaxPoolModuleMapper],
    "paddle.nn.MaxPool3D": ["tensorlayerx.nn.MaxPool3d", MaxPoolModuleMapper],
    # "paddle.nn.Upsample": ["tensorlayerx.nn.UpSampling2d", UpSamplingModuleMapper],
    # "paddle.nn.Pad2D": ["tensorlayerx.nn.ZeroPad2d", None],
    "paddle.nn.MultiHeadAttention": ["tensorlayerx.nn.MultiheadAttention", MultiHeadAttentionMapper],  # nlp task
    "paddle.nn.LSTM": ["tensorlayerx.nn.LSTM", LstmMapper],
    "paddle.nn.SimpleRNN": ["tensorlayerx.nn.RNN", RnnMapper],
    "paddle.flatten": ["tensorlayerx.flatten", FlattenOpMapper],
    "paddle.nn.Flatten": ["tensorlayerx.nn.Flatten", FlattenModuleMapper],

    # act_layer - nn
    # "paddle.nn.GELU": ["tensorlayerx.ops.gelu", None],
    "paddle.nn.Hardswish": ["tensorlayerx.nn.Hardswish", None],  # 0.5.7 support
    "paddle.nn.Hardsigmoid": ["tensorlayerx.nn.HardSigmoid", None],
    "paddle.nn.LeakyReLU": ["tensorlayerx.nn.LeakyReLU", None],
    "paddle.nn.PReLU": ["tensorlayerx.nn.PRelu", None],  # dsifn - tensorlayerx.layers.PRelu
    "paddle.nn.ReLU": ["tensorlayerx.nn.ReLU", None],
    "paddle.nn.ReLU6": ["tensorlayerx.nn.ReLU6", None],
    "paddle.nn.Sigmoid": ["tensorlayerx.nn.Sigmoid", None],
    "paddle.nn.Softmax": ["tensorlayerx.nn.Softmax", None],
    "paddle.nn.Tanh": ["tensorlayerx.nn.Tanh", None],
    "paddle.nn.Swish": ["tensorlayerx.nn.Swish", None],

    # act_layer - functional
    # "paddle.nn.functional.avg_pool1d": ["tensorlayerx.ops.AvgPool1d", None],
    # "paddle.nn.functional.avg_pool3d": ["tensorlayerx.ops.AvgPool3d", None],
    "paddle.nn.functional.dropout": ["tensorlayerx.ops.Dropout", DropoutModuleMapper],
    "paddle.nn.functional.leaky_relu": ["tensorlayerx.ops.leaky_relu", None],
    # # "paddle.nn.functional.pad": [" tensorlayerx.vision.transforms.pad", None],
    "paddle.nn.functional.relu": ["tensorlayerx.ops.relu", None],
    # "paddle.nn.functional.gelu": ["tensorlayerx.ops.gelu", None],
    "paddle.nn.functional.sigmoid": ["tensorlayerx.ops.sigmoid", None],
    "paddle.nn.functional.softmax": ["tensorlayerx.ops.softmax", None],
    "paddle.nn.functional.swish": ["tensorlayerx.ops.swish", None],  # tensorlayerx.nn.Swish
    # "paddle.tanh": ["tensorlayerx.ops.tanh", None],
    "paddle.nn.functional.relu6": ["tensorlayerx.nn.ReLU6()", None],
    "paddle.nn.functional.hardswish": ["tensorlayerx.ops.hardswish", None],  # todo

    # nn - functional
    # "paddle.nn.functional.linear": ["tensorlayerx.ops.matmul", LinearOpMapper],
    # "paddle.nn.functional.linear": ["tensorlayerx.ops.linear", LinearOpMapper],  # log-20230106 - gan
}


INIT_MAPPER = {
    # special case
    "paddle.fluid.param_attr.ParamAttr": ["tensorlayerx.nn.initializers.random_uniform", ParamAttrRandomMapper],
    "paddle.ParamAttr": ["tensorlayerx.nn.initializers.xavier_uniform", ParamAttrXavierMapper],

    # init
    "paddle.nn.initializer.Uniform": ["tensorlayerx.nn.initializers.random_uniform", None],
    "paddle.nn.initializer.TruncatedNormal": ["tensorlayerx.nn.initializers.TruncatedNormal", TruncatedNormalOpMapper],
    "paddle.nn.initializer.KaimingNormal": ["tensorlayerx.nn.initializers.HeNormal", None],
    "paddle.nn.initializer.KaimingUniform": ["tensorlayerx.nn.initializers.HeUniform", None],
    "paddle.nn.initializer.XavierNormal": ["tensorlayerx.nn.initializers.XavierNormal", None],
    "paddle.nn.initializer.XavierUniform": ["tensorlayerx.nn.initializers.XavierUniform", None],
    "paddle.nn.initializer.Constant": ["tensorlayerx.nn.initializers.Constant", None],
    # "paddle.nn.initializer.Normal": ["tensorlayerx.nn.initializers.RandomNormal", None],  #
    "paddle.nn.initializer.Normal": ["tensorlayerx.nn.initializers.random_normal", RandomNormalMapper],
    "paddle.ones": ["tensorlayerx.ones", None],
    "paddle.zeros": ["tensorlayerx.zeros", None],
}


LOSS_MAPPER = {
    # loss - nn
    "paddle.nn.CrossEntropyLoss": ["paddle.nn.CrossEntropyLoss", None],
    # "paddle.nn.CrossEntropyLoss": ["tensorlayerx.losses.softmax_cross_entropy_with_logits", None],
    "paddle.nn.BCELoss": ["tensorlayerx.losses.binary_cross_entropy", None],
    "paddle.nn.BCEWithLogitsLoss": ["tensorlayerx.losses.binary_cross_entropy", None],
    "paddle.nn.functional.binary_cross_entropy": ["tensorlayerx.losses.binary_cross_entropy", None],

    # loss - functional
    # "paddle.nn.functional.binary_cross_entropy_with_logits": ["tensorlayerx.losses.binary_cross_entropy", None],
    "paddle.nn.functional.binary_cross_entropy_with_logits": ["tensorlayerx.losses.sigmoid_cross_entropy",
                                                              BCEwithLogitLossMapper],  # log-20221226
    "paddle.nn.functional.mse_loss": ["tensorlayerx.losses.mean_squared_error", None],
    "paddle.nn.functional.cosine_similarity": ["tensorlayerx.losses.cosine_similarity", None],
    "paddle.nn.functional.sigmoid_focal_loss": ["tensorlayerx.losses.sigmoid_cross_entropy",
                                                SigmoidCrossEntropyOpMapper],
}


METRIC_MAPPER = {
    # metric
    "paddle.metric.accuracy": ["paddle.metric.accuracy", None],
    # "paddle.nn.functional.cross_entropy": ["paddle.nn.functional.cross_entropy", None]
}


OPTIMIZER_MAPPER = {
    # "paddle.optimizer": ["tensorlayerx.optimizers", None],
    "paddle.optimizer": ["paddle.optimizer", None],
    "paddle.optimizer.lr.ReduceOnPlateau": ["tensorlayerx.optimizers.lr.ReduceOnPlateau", None],
    "paddle.optimizer.lr.CosineAnnealingDecay": ["tensorlayerx.optimizers.lr.CosineAnnealingDecay", None],
    "paddle.optimizer.lr.MultiStepDecay": ["tensorlayerx.optimizers.lr.MultiStepDecay", None],
    "paddle.optimizer.Adam": ["paddle.optimizer.Adam", None],
    # "paddle.optimizer.Adam": ["tensorlayerx.optimizers.Adam", AdamModuleMapper],
    # "paddle.optimizer.Momentum": ["tensorlayerx.optimizers.Momentum", None],
    "paddle.optimizer.SGD": ["tensorlayerx.optimizers.SGD", None],
    "paddle.optimizer.lr.LinearWarmup": ["tensorlayerx.optimizers.lr.LinearWarmup", None],
    "paddle.optimizer.lr.LRScheduler": ["tensorlayerx.optimizers.lr.LRScheduler", None],
    # "paddle.optimizer.lr.PolynomialDecay": ["tensorlayerx.optimizers.lr.PolynomialDecay", None],
    "paddle.optimizer.lr.PiecewiseDecay": ["tensorlayerx.optimizers.lr.PiecewiseDecay", None],
    "paddle.optimizer.lr.StepDecay": ["tensorlayerx.optimizers.lr.StepDecay", None],
    "paddle.optimizer.Optimizer": ["tensorlayerx.optimizers.paddle_optimizers.Optimizer", None],
}


UTILS_MAPPER = {
    "paddle.io.Dataset": ["tensorlayerx.dataflow.Dataset", None],
    # "paddle.io.DataLoader": ["tensorlayerx.dataflow.DataLoader", DataLoaderMapper],
    "paddle.io.IterableDataset": ["tensorlayerx.dataflow.IterableDataset", None],
    "paddle.io.BatchSampler": ["tensorlayerx.dataflow.BatchSampler", BatchSamplerModuleMapper],
    # "paddle.io.DistributedBatchSampler": ["tensorlayerx.dataflow.BatchSampler", DistributedBatchSamplerMapper],
    "paddle.io.random_split": ["tensorlayerx.dataflow.random_split", None],
    "paddle.get_device": ["tensorlayerx.ops.get_device", None],
    "paddle.nn.ClipGradByGlobalNorm": ["tensorlayerx.ClipByGlobalNorm", None],
}


VISION_MAPPER = {
    "paddle.vision": ["tensorlayerx.vision", None],
    # data enhancement & utils
    "paddle.vision.transforms": ["tensorlayerx.vision.transforms", None],
    "paddle.vision.transforms.adjust_brightness": ["tensorlayerx.vision.transforms.AdjustBrightness", None],
    "paddle.vision.transforms.adjust_contrast": ["tensorlayerx.vision.transforms.AdjustContrast", None],
    "paddle.vision.transforms.adjust_hue": ["tensorlayerx.vision.transforms.AdjustHue", None],
    "paddle.vision.transforms.ColorJitter": ["tensorlayerx.vision.transforms.ColorJitter", None],
    "paddle.vision.transforms.Compose": ["tensorlayerx.vision.transforms.Compose", None],
    "paddle.vision.transforms.crop": ["tensorlayerx.vision.transforms.Crop", None],
    "paddle.vision.transforms.functional.crop": ["tensorlayerx.vision.transforms.Crop", None],
    "paddle.vision.transforms.center_crop": ["tensorlayerx.vision.transforms.CentralCrop", None],
    "paddle.vision.transforms.hflip": ["tensorlayerx.vision.transforms.FlipHorizontal", None],
    "paddle.vision.transforms.normalize": ["tensorlayerx.vision.transforms.Normalize", None],
    "paddle.vision.transforms.pad": ["tensorlayerx.vision.transforms.Pad", None],
    "paddle.vision.transforms.RandomCrop": ["tensorlayerx.vision.transforms.RandomCrop", None],
    "paddle.vision.transforms.RandomHorizontalFlip": ["tensorlayerx.vision.transforms.RandomFlipHorizontal", None],
    "paddle.vision.transforms.RandomResizedCrop": ["tensorlayerx.vision.transforms.RandomResizedCrop", None],
    "paddle.vision.transforms.RandomRotate": ["tensorlayerx.vision.transforms.RandomRotation", None],
    "paddle.vision.transforms.RandomVerticalFlip": ["tensorlayerx.vision.transforms.RandomFlipVertical", None],
    "paddle.vision.transforms.Resize": ["tensorlayerx.vision.transforms.Resize", None],
    "paddle.vision.transforms.rotate": ["tensorlayerx.vision.transforms.Rotation", None],
    "paddle.vision.transforms.SaturationTransform": ["tensorlayerx.vision.transforms.AdjustSaturation", None],
    "paddle.vision.transforms.to_grayscale": ["tensorlayerx.vision.transforms.RgbToGray", None],
    "paddle.vision.transforms.ToTensor": ["tensorlayerx.vision.transforms.ToTensor", None],  # CHW -> HWC
    "paddle.vision.transforms.Transpose": ["tensorlayerx.vision.transforms.Transpose", None],
    "paddle.vision.transforms.vflip": ["tensorlayerx.vision.transforms.FlipVertical", None],

    # load
    "paddle.vision.image.image_load": ["tensorlayerx.vision.utils.load_image", None],
    # "": ["tensorlayerx.vision.utils.load_images", None],
    # "": ["tensorlayerx.vision.utils.save_images", None],
    # metric
    # "": ["tensorlayerx.vision.nms", None],
    # "": ["tensorlayerx.vision.box_iou", None],
    # "": ["tensorlayerx.vision.box_area", None],
}


API_MAPPER = {
    "paddle": ["tensorlayerx", None],
    # "paddle.load": ["paddle.load", None],
    # "paddle.load": ["tensorlayerx.files.load_npz", None],  # load_npz - load_and_assign_npz_dict
    # "paddle.save": ["tensorlayerx.files.save_npz", None],  # save_npz - save_npz_dict
    "paddle.set_device": ["tensorlayerx.set_device", None],
    "paddle.device.get_device": ["tensorlayerx.get_device", None],
    # "paddle.": ["tensorlayerx.to_device", None],
    "paddle.concat": ["tensorlayerx.concat", ConcatOpMapper],
    "paddle.to_tensor": ["tensorlayerx.convert_to_tensor", None],
    "paddle.seed": ["tensorlayerx.set_seed", None],
    "paddle.unsqueeze": ["tensorlayerx.expand_dims", None],  # .ops is also ok
    "paddle.tensor.unsqueeze": ["tensorlayerx.ops.expand_dims", None],
    "paddle.squeeze": ["tensorlayerx.ops.squeeze", None],
    "paddle.transpose": ["tensorlayerx.transpose", None],
    "paddle.reshape": ["tensorlayerx.reshape", ReshapeMapper],  # don't import ops
    "paddle.cumsum": ["tensorlayerx.cumsum", CumsumOpMapper],
    "paddle.sum": ["tensorlayerx.reduce_sum", SumOpMapper],  # log-20221226
    "paddle.mean": ["tensorlayerx.reduce_mean", MeanOpMapper],
    "paddle.ones": ["tensorlayerx.ops.ones", None],
    "paddle.zeros": ["tensorlayerx.ops.zeros", None],
    "paddle.sqrt": ["tensorlayerx.ops.sqrt", None],
    "paddle.arange": ["tensorlayerx.ops.arange", ArangeOpMapper],
    "paddle.matmul": ["tensorlayerx.ops.matmul", MatMulOpMapper],
    "paddle.exp": ["tensorlayerx.ops.exp", None],
    "paddle.max": ["tensorlayerx.reduce_max", MaxOpMapper],  # .ops.xxx
    "paddle.min": ["tensorlayerx.reduce_min", None],
    "paddle.argmax": ["tensorlayerx.ops.argmax", ArgmaxOpMapper],
    "paddle.argmin": ["tensorlayerx.ops.argmin", None],
    "paddle.stack": ["tensorlayerx.ops.stack", StackOpMapper],
    "paddle.log": ["tensorlayerx.ops.log", None],
    "paddle.abs": ["tensorlayerx.ops.abs", None],
    "paddle.logical_or": ["tensorlayerx.ops.logical_or", None],
    "paddle.logical_xor": ["tensorlayerx.ops.logical_xor", None],
    "paddle.logical_and": ["tensorlayerx.ops.logical_and", None],
    "paddle.logical_not": ["tensorlayerx.ops.logical_not", None],
    "paddle.split": ["tensorlayerx.ops.split", SplitOpMapper],
    "paddle.add": ["tensorlayerx.add", AddOpMapper],  # note: tensorlayerx.ops.add is not support
    "paddle.multiply": ["tensorlayerx.ops.multiply", None],
    "paddle.einsum": ["tensorlayerx.ops.einsum", None],
    "paddle.floor": ["tensorlayerx.floor", None],
    # "paddle.shape": ["tensorlayerx.ops.get_tensor_shape", None],
    "paddle.rand": ["tensorlayerx.ops.random_uniform", None],
    "paddle.clip": ["tensorlayerx.ops.clip_by_value", ClipOpMapper],
    "paddle.add_n": ["tensorlayerx.ops.add_n", None],
    "paddle.tensor.transpose": ["tensorlayerx.ops.transpose", None],
    # "paddle.linspace": ["tensorlayerx.linspace", None],
    "paddle.roll": ["tensorlayerx.roll", RollOpMapper],
    "paddle.meshgrid": ["tensorlayerx.meshgrid", None],
    "paddle.mm": ["tensorlayerx.matmul", None],
    "paddle.gather": ["tensorlayerx.gather", GatherOpMapper],  # .ops
    "paddle.cast": ["tensorlayerx.cast", None],
    "paddle.tile": ["tensorlayerx.tile", None],
    "paddle.subtract": ["tensorlayerx.subtract", None],
    "paddle.divide": ["tensorlayerx.divide", None],
    "paddle.topk": ["tensorlayerx.ops.topk", None],
    "paddle.ones_like": ["tensorlayerx.ones_like", None],
    "paddle.nn.functional.one_hot": ["tensorlayerx.ops.OneHot", None],
    "paddle.equal": ["tensorlayerx.ops.equal", None],
    "paddle.where": ["tensorlayerx.where", None],
    "paddle.full": ["tensorlayerx.constant", FullOpMapper],
    "paddle.full_like": ["tensorlayerx.constant", FullLikeOpMapper],
    "paddle.zeros_like": ["tensorlayerx.zeros_like", None],
    # "paddle.nonzero": ["tensorlayerx.CountNonzero()", None],
    "paddle.gather_nd": ["tensorlayerx.gather_nd", None],
    "paddle.maximum": ["tensorlayerx.maximum", None],
    "paddle.minimum": ["tensorlayerx.minimum", None],
    "paddle.masked_select": ["tensorlayerx.mask_select", None],
    "paddle.tensor.meshgrid": ["tensorlayerx.meshgrid", None],
    "paddle.diag": ["tensorlayerx.diag", None],
    "paddle.framework.get_default_dtype": ["tensorlayerx.ops.paddle_nn.framework.get_default_dtype", None],
    "paddle.triu": ["tensorlayerx.triu", None],
    "paddle.argsort": ["tensorlayerx.argsort", None],
    "paddle.floor_divide": ["tensorlayerx.floordiv", None],
    "paddle.pow": ["tensorlayerx.pow", None],
    "paddle.cos": ["tensorlayerx.cos", None],
    "paddle.chunk": ["tensorlayerx.split", chunkOpMapper],
    "paddle.bmm": ["tensorlayerx.bmm", None],
    "paddle.randn": ["tensorlayerx.ops.random_normal", None],  # log 20221226
    "paddle.rsqrt": ["tensorlayerx.rsqrt", None],  # log 20221226
    "paddle.var": ["tensorlayerx.reduce_variance", varOpMapper],  # log 20221226
    "paddle.flip": ["tensorlayerx.flip", None],  # log 20221226
    "paddle.static.nn.fc": ["paddle.static.nn.fc", None],  # log 20221226
    "paddle.nn.functional.conv2d_transpose": ["paddle.nn.functional.conv2d_transpose", None],  # log 20221226
    "paddle.numel": ["tensorlayerx.numel", None],  # log 20221226
    "paddle.greater_than": ["tensorlayerx.greater", None],  # log 20221226
    "paddle.atan": ["tensorlayerx.atan", None],  # log 20221226
    "paddle.index_select": ["tensorlayerx.index_select", None],  # log-20221226
}

CUSTOM_API = {
    # paddle2tlx api utils

    # nn
    "paddle.nn.Dropout": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout", None],
    "paddle.nn.Dropout2D": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout", None],
    "paddle.nn.AvgPool2D": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_AvgPool2d", None],
    "paddle.nn.Upsample": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_Upsample", UpsampleModuleMapper],
    "paddle.nn.MaxUnPool2D": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxUnPool2d", MaxUnPool2DModuleMapper],
    "paddle.nn.Identity": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_Identity", None],
    "paddle.nn.InstanceNorm1D": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_InstanceNorm1d", InstanceNormModuleMapper], # log-20221226
    "paddle.nn.InstanceNorm2D": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_InstanceNorm2d", InstanceNormModuleMapper], # log-20221226
    "paddle.nn.InstanceNorm3D": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_InstanceNorm3d", InstanceNormModuleMapper], # log-20221226
    "paddle.nn.UpsamplingBilinear2D": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_UpsamplingBilinear2d", None],
    "paddle.nn.Pad2D": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d", Pad2DModuleMapper],  # log-20221226 tlx_Pad2d
    "paddle.nn.GELU": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_GELU", None],
    "paddle.nn.MaxPool2D": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d", None],
    "paddle.nn.Conv2DTranspose": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_ConvTranspose2d", Conv2DTransposeMapper],  # fcef, dsamnet
    "paddle.vision.ops.DeformConv2D": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_DeformConv2d", DeformConv2DMapper],  # log-20221226

    # dataflow utils
    "paddle.load": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_load", None],  # TODO
    "paddle2tlx.pd2tlx.utils.load_model_clas": ["paddle2tlx.pd2tlx.utils.restore_model_clas", None],
    "paddle2tlx.pd2tlx.utils.load_model_nlp": ["paddle2tlx.pd2tlx.utils.restore_model_nlp", None],
    "paddle2tlx.pd2tlx.utils.load_model_seg": ["paddle2tlx.pd2tlx.utils.restore_model_seg", None],
    "paddle2tlx.pd2tlx.utils.load_model_rsseg": ["paddle2tlx.pd2tlx.utils.restore_model_rsseg", None],
    "paddle2tlx.pd2tlx.utils.load_model_det": ["paddle2tlx.pd2tlx.utils.restore_model_det", None],
    "paddle2tlx.pd2tlx.utils.load_model_cdet": ["paddle2tlx.pd2tlx.utils.restore_model_cdet", None],
    "paddle2tlx.pd2tlx.utils.load_model_gan": ["paddle2tlx.pd2tlx.utils.restore_model_gan", None],

    # ops
    "paddle.linspace": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_linspace", None],
    "paddle.nonzero": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_nonzero", None],   # log-20221227
    "paddle.nn.functional.linear": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_linear", None],  # log-20230112 - det
    "partial": ["partial", PartialTLXMapper],  # TODO
    "paddle.shape": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape", None],
    "paddle.nn.layer.activation.__dict__.keys": ["tensorlayerx.nn.layers.activation.__dict__.keys", None],
    "paddle.optimizer.lr": ["tensorlayerx.optimizers.lr", None],
    # "norm_layer": ["norm_layer", BatchNormModuleMapper],
    "self._batch_norm": ["self._batch_norm", BatchNormModuleMapper],
    "self.batch_norm": ["self.batch_norm", None],  # log-20230104-seg-需注释
    "batch_norm": ["batch_norm", BatchNormModuleMapper],
    "layer_norm": ["layer_norm", LayerNormModuleMapper],
    # "act_layer": ["act_layer", CustomFuncActMapper],
    "model.eval": ["model.set_eval", None],
    "self.encoder1.eval": ["self.encoder1.set_eval", None],  # dsifn
    "self.encoder2.eval": ["self.encoder2.set_eval", None],  # dsifn

    # losses
    "paddle.nn.BCEWithLogitsLoss": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_BCEWithLogitsLoss", None],  # log-20221226
    "paddle.nn.MSELoss": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_MSELoss", None],  # log-20221226
    "paddle.nn.L1Loss": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_L1Loss", None],    # log-20221226
    "paddle.nn.functional.L1Loss": ["tensorlayerx.losses.L1Loss", L1LossOpMapper],  # log-20221226

    # metric
    "paddle.nn.functional.cross_entropy": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_cross_entropy", None],

    # others - need optimize
    "self.create_parameter": ["self.create_parameter", CreateParameterMapper],
    "self._set_eval_pd": ["self._set_eval_tlx", None],
    "self.forward_pd": ["self.forward_tlx", None],
    "self.bottlenecks_layer_pd": ["self.bottlenecks_layer_tlx", None],
    "self.blocks2_layer_pd": ["self.blocks2_layer_tlx", None],
    "self.save_model_pd": ["self.save_model_tlx", None],
    "self.init_weight_pd": ["self.init_weight_tlx", None],
    "self.get_sequential_pd": ["self.get_sequential_tlx", None],
    "self.init_pd": ["self.init_tlx", None],

    # examples
    "paddle2tlx.examples.predict_clas.predict_pd": ["paddle2tlx.examples.predict_clas.predict_tlx", None],
    "paddle2tlx.examples.predict_det.predict_pd": ["paddle2tlx.examples.predict_det.predict_tlx", None],
    "paddle2tlx.examples.predict_cdet.predict_pd": ["paddle2tlx.examples.predict_cdet.predict_tlx", None],
    "paddle2tlx.examples.predict_seg.predict_pd": ["paddle2tlx.examples.predict_seg.predict_tlx", None],
    "paddle2tlx.pd2tlx.utils.eval_init_pd": ["paddle2tlx.pd2tlx.utils.eval_init_tlx", None],
}

HOLDON_API = {
    # the follow api tlx is not support, so use paddle api

    # nn layer
    "paddle.nn.GroupNorm": ["paddle.nn.GroupNorm", None],

    # init
    "paddle.regularizer.L2Decay": ["paddle.regularizer.L2Decay", None],

    # dataflow utils
    "paddle.utils.download.get_weights_path_from_url": ["paddle.utils.download.get_weights_path_from_url", None],
    "paddle.save": ["paddle.save", None],
    "paddle.io.DataLoader": ["paddle.io.DataLoader", None],
    "paddle.io.DistributedBatchSampler": ["paddle.io.DistributedBatchSampler", None],

    # common
    # "paddle.index_select": ["paddle.index_select", None],
    "paddle.Tensor": ["paddle.Tensor", None],
    # "paddle.full_like": ["paddle.full_like", None],
    "paddle.randperm": ["paddle.randperm", None],
    "paddle.unbind": ["paddle.unbind", None],
    "paddle.unique": ["paddle.unique", None],
    "paddle.histogram": ["paddle.histogram", None],
    "paddle.utils.cpp_extension.setup": ["paddle.utils.cpp_extension.setup", None],
    "paddle.utils.cpp_extension.CUDAExtension": ["paddle.utils.cpp_extension.CUDAExtension", None],
    "paddle.device.is_compiled_with_cuda": ["paddle.device.is_compiled_with_cuda", None],
    "paddle.utils.cpp_extension.CppExtension": ["paddle.utils.cpp_extension.CppExtension", None],
    "paddle.index_sample": ["paddle.index_sample", None],
    "paddle.expand_as": ["paddle.expand_as", None],
    "paddle.expand": ["paddle.expand", None],
    "paddle.vision.ops.roi_align": ["paddle.vision.ops.roi_align", None],
    "paddle.vision.ops.distribute_fpn_proposals": ["paddle.vision.ops.distribute_fpn_proposals", None],
    "paddle.vision.ops": ["paddle.vision.ops", None],
    "paddle.slice": ["paddle.slice", None],
    "paddle.vision.ops.yolo_box": ["paddle.vision.ops.yolo_box", None],
    "paddle.version": ["paddle.version", None],
    "paddle.version.major": ["paddle.version.major", None],
    "paddle.version.minor": ["paddle.version.minor", None],
    "paddle.scatter": ["tensorlayerx.scatter_update", None],
    "paddle.unsqueeze_": ["paddle.unsqueeze_", None],
    "paddle.uniform": ["paddle.uniform", None],
    "paddle.normal": ["paddle.normal", None],
    "paddle.fluid.dataloader.collate.default_collate_fn": ["paddle.fluid.dataloader.collate.default_collate_fn", None],
    "paddle.distributed.get_world_size": ["paddle.distributed.get_world_size", None],
    "paddle.distributed.get_rank": ["paddle.distributed.get_rank", None],
    "paddle.distributed.init_parallel_env": ["paddle.distributed.init_parallel_env", None],
    "paddle.DataParallel": ["paddle.DataParallel", None],
    "paddle.distributed.ParallelEnv": ["paddle.distributed.ParallelEnv", None],
    "paddle.get_cudnn_version": ["paddle.get_cudnn_version", None],

    # gradient
    "paddle.no_grad": ["paddle.no_grad", None],  # todo

    # functional
    # "paddle.nn.functional": ["paddle.nn.functional", None],
    # "paddle.nn.functional.sigmoid": ["paddle.nn.functional.sigmoid", None],
    # "paddle.nn.functional.swish": ["paddle.nn.functional.swish", None],
    "paddle.nn.functional.conv2d": ["paddle.nn.functional.conv2d", None],
    "paddle.nn.functional.grid_sample": ["paddle.nn.functional.grid_sample", None],
    "paddle.nn.functional.interpolate": ["paddle.nn.functional.interpolate", None],
    "paddle.nn.functional.max_unpool2d": ["paddle.nn.functional.max_unpool2d", None],
    "paddle.nn.functional.normalize": ["paddle.nn.functional.normalize", None],
    "paddle.nn.functional.avg_pool2d": ["paddle.nn.functional.avg_pool2d", None],
    "paddle.nn.functional.max_pool2d": ["paddle.nn.functional.max_pool2d", None],
    "paddle.nn.functional.adaptive_avg_pool2d": ["paddle.nn.functional.adaptive_avg_pool2d", None],
    "paddle.nn.functional.l1_loss": ["paddle.nn.functional.l1_loss", None],
    "paddle.nn.functional.sigmoid_focal_loss": ["paddle.nn.functional.sigmoid_focal_loss", None],
    "paddle.nn.functional.smooth_l1_loss": ["paddle.nn.functional.smooth_l1_loss", None],
    # "paddle.nn.functional.adaptive_max_pool2d": ["paddle2tlx.pd2tlx.ops.tlxops.tlx_adaptive_max_pool2d", None], # log-2023-0103
    "paddle.nn.functional.adaptive_max_pool2d": ["paddle.nn.functional.adaptive_max_pool2d", AdaptionMaxPool2dOpMapper],
    "paddle.nn.functional.pad": ["paddle.nn.functional.pad", None],
    "paddle.nn.functional.unfold": ["paddle.nn.functional.unfold", None],
    "paddle.nn.functional.hardswish": ["paddle.nn.functional.hardswish", None],

    # paddle related
    "paddle.common_ops_import.check_type": ["paddle.common_ops_import.check_type", None],
    "paddle.nn.functional.silu": ["paddle.nn.functional.silu", None],
    "paddle.common_ops_import.check_variable_and_dtype": ["paddle.common_ops_import.check_variable_and_dtype", None],
    "paddle.nn.functional.mish": ["paddle.nn.functional.mish", None],  # TODO
    "paddle.in_dynamic_mode": ["paddle.in_dynamic_mode", None],
    "paddle._C_ops": ["paddle._C_ops", None],
    "paddle._legacy_C_ops": ["paddle._legacy_C_ops", None],
    "paddle.common_ops_import.LayerHelper": ["paddle.common_ops_import.LayerHelper", None],
    "paddle.common_ops_import.Variable": ["paddle.common_ops_import.Variable", None],
    "paddle.vision.ops.generate_proposals": ["paddle.vision.ops.generate_proposals", None],
    "paddle.version.rc": ["paddle.version.rc", None],
    "paddle.is_compiled_with_npu": ["paddle.is_compiled_with_npu", None],
    "paddle.is_compiled_with_xpu": ["paddle.is_compiled_with_xpu", None],
    "paddle.is_compiled_with_cuda": ["paddle.is_compiled_with_cuda", None],
    "paddle.version.patch": ["paddle.version.patch", None],
    "paddle._legacy_C_ops.argsort": ["paddle._legacy_C_ops.argsort", None],
    "paddle._C_ops.argsort": ["paddle._C_ops.argsort", None],
    "paddle._C_ops.multiclass_nms3": ["paddle._C_ops.multiclass_nms3", None],
    "paddle._C_ops.generate_proposals_v2": ["paddle._C_ops.generate_proposals_v2", None],
    "paddle._C_ops.box_coder": ["paddle._C_ops.box_coder", None],
    "paddle._C_ops.matrix_nms": ["paddle._C_ops.matrix_nms", None],
    "paddle._C_ops.prior_box": ["paddle._C_ops.prior_box", None],
    "paddle._C_ops.distribute_fpn_proposals": ["paddle._C_ops.distribute_fpn_proposals", None],

    # opt
    "paddle.optimizer.lr.PolynomialDecay": ["paddle.optimizer.lr.PolynomialDecay", None],
    "paddle.optimizer.Momentum": ["paddle.optimizer.Momentum", None],
}


DATATYPE = {
    "paddle.bool": ["tensorlayerx.bool", None],
    "paddle.int8": ["tensorlayerx.int8", None],
    "paddle.int16": ["tensorlayerx.int16", None],
    "paddle.int32": ["tensorlayerx.int32", None],
    "paddle.int64": ["tensorlayerx.int64", None],
    "paddle.uint8": ["tensorlayerx.uint8", None],
    "paddle.uint16": ["tensorlayerx.uint16", None],
    "paddle.uint32": ["tensorlayerx.uint32", None],
    "paddle.uint64": ["tensorlayerx.uint64", None],
    "paddle.float16": ["tensorlayerx.float16", None],
    "paddle.float32": ["tensorlayerx.float32", None],
    "paddle.float64": ["tensorlayerx.float64", None],
}


API_MAPPER.update(NN_MAPPER)
API_MAPPER.update(INIT_MAPPER)
API_MAPPER.update(LOSS_MAPPER)
API_MAPPER.update(METRIC_MAPPER)
API_MAPPER.update(OPTIMIZER_MAPPER)
API_MAPPER.update(UTILS_MAPPER)
API_MAPPER.update(VISION_MAPPER)
API_MAPPER.update(CUSTOM_API)
API_MAPPER.update(HOLDON_API)
API_MAPPER.update(DATATYPE)


# literal name replace
REPLACE_API = {
    # when this op as parameter assignment
    "ParamAttr(initializer=Uniform(-stdv, stdv))": "random_uniform(-stdv, stdv)",  # alexnet
    "ParamAttr": "tensorlayerx.nn.initializers.xavier_uniform",  # regnet
    "BatchNorm2D": "BatchNorm2d",
    "nn.BatchNorm2D": "nn.BatchNorm2d",
    "nn.Hardsigmoid": "nn.HardSigmoid",
    "nn.GELU": "tensorlayerx.ops.GeLU",  # tensorlayerx.ops.gelu
    "nn.Layer": "nn.Mudule",
    # "paddle.optimizer.lr": "tensorlayerx.optimizers.lr",
    "Normal": "tensorlayerx.nn.initializers.random_normal",  # TODO
    "NCHW": "channels_first",  # log-20221226-27-seg
    "NHWC": "channels_last",   # log-20221226-27-seg
}

HEADER_IMPORT = {
    "__future__",
}

NOT_SUPPORT_API = {
    "paddle.nn.SyncBatchNorm",
}


ATTRIBUTE_MAPPING = {
    # attribute
    "_kernel_size": "kernel_size",
    "_out_channels": "out_channels",
}


REMOVE_API = [
    # paddle api
    "paddle.fluid.io.shuffle",
    "paddle.fluid.layers.adaptive_pool2d",
    "paddle.fluid.layers.zeros_like",
    "paddle.fluid.layers.hard_swish",
    "paddle.fluid.metrics.Accuracy",
    "paddle.fluid.layers.one_hot",
    "paddle.fluid.optimizer.SGDOptimizer",
    "paddle.regularizer.L2Decay",
    "paddle.fluid.param_attr.ParamAttr",
    "paddle.nn.functional",
]
