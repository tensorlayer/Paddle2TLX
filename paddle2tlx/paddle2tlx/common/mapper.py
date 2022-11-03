# coding: utf-8
from paddle2tlx.api_mapper import *


NN_MAPPER = {
    # basic
    "paddle.nn": ["tensorlayerx.nn", None],
    "paddle.nn.Layer": ["tensorlayerx.nn.Module", None],
    "paddle.nn.LayerList": ["tensorlayerx.nn.ModuleList", None],
    "paddle.nn.Sequential": ["tensorlayerx.nn.Sequential", None],
    "paddle.create_parameter": ["tensorlayerx.nn.Parameter", None],

    # nn_net
    "paddle.nn.AdaptiveAvgPool1D": ["tensorlayerx.nn.AdaptiveAvgPool1d", AdaptiveAvgPoolMapper],
    "paddle.nn.AdaptiveAvgPool2D": ["tensorlayerx.nn.AdaptiveAvgPool2d", AdaptiveAvgPoolMapper],
    "paddle.nn.AdaptiveAvgPool3D": ["tensorlayerx.nn.AdaptiveAvgPool3d", AdaptiveAvgPoolMapper],
    "paddle.nn.AvgPool1D": ["tensorlayerx.nn.AvgPool1d", AvgPoolFuncMapper],
    "paddle.nn.AvgPool2D": ["tensorlayerx.nn.AvgPool2d", AvgPoolFuncMapper],
    "paddle.nn.AvgPool3D": ["tensorlayerx.nn.AvgPool3d", AvgPoolFuncMapper],
    "paddle.nn.BatchNorm": ["tensorlayerx.nn.BatchNorm", BatchNormModuleMapper],
    "paddle.nn.BatchNorm1D": ["tensorlayerx.nn.BatchNorm1d", BatchNormModuleMapper],
    "paddle.nn.BatchNorm2D": ["tensorlayerx.nn.BatchNorm2d", BatchNormModuleMapper],
    "paddle.nn.BatchNorm3D": ["tensorlayerx.nn.BatchNorm3d", BatchNormModuleMapper],
    # "paddle.nn.Conv1D": ["tensorlayerx.nn.Conv1d", ConvModuleMapper],  # TODO
    # "paddle.nn.Conv2D": ["tensorlayerx.nn.Conv2d", ConvModuleMapper],
    # "paddle.nn.Conv3D": ["tensorlayerx.nn.Conv3d", ConvModuleMapper],
    "paddle.nn.Conv1D": ["tensorlayerx.nn.GroupConv1d", GroupConvModuleMapper],  # mobilenet
    "paddle.nn.Conv2D": ["tensorlayerx.nn.GroupConv2d", GroupConvModuleMapper],
    "paddle.nn.Conv3D": ["tensorlayerx.nn.GroupConv3d", GroupConvModuleMapper],
    "paddle.nn.Conv2DTranspose": ["tensorlayerx.nn.ConvTranspose2d", None],
    "paddle.nn.Dropout": ["tensorlayerx.nn.Dropout", DropoutModuleMapper],
    "paddle.nn.Embedding": ["tensorlayerx.nn.Embedding", None],
    "paddle.nn.LayerNorm": ["tensorlayerx.nn.LayerNorm", LayerNormModuleMapper],
    "paddle.nn.Linear": ["tensorlayerx.nn.Linear", LinearModuleMapper],
    "paddle.nn.MaxPool1D": ["tensorlayerx.nn.MaxPool1d", MaxPoolModuleMapper],
    "paddle.nn.MaxPool2D": ["tensorlayerx.nn.MaxPool2d", MaxPoolModuleMapper],
    "paddle.nn.MaxPool3D": ["tensorlayerx.nn.MaxPool3d", MaxPoolModuleMapper],
    "paddle.nn.Upsample": ["tensorlayerx.nn.UpSampling2d", None],  # TODO - 区别待确认
    "paddle.nn.Pad2D": ["tensorlayerx.nn.ZeroPad2d", None],  # TODO
    "paddle.flatten": ["tensorlayerx.nn.Flatten()", FlattenModuleMapper],  # TODO

    # act_layer - nn
    "paddle.nn.GELU": ["tensorlayerx.ops.gelu", None],  # TODO
    "paddle.nn.Hardswish": ["tensorlayerx.nn.Hardswish", None],  # 0.5.7 support
    "paddle.nn.Hardsigmoid": ["tensorlayerx.nn.HardSigmoid", None],
    "paddle.nn.LeakyReLU": ["tensorlayerx.nn.LeakyReLU", None],
    "paddle.nn.PReLU": ["tensorlayerx.nn.PReLU", None],
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
    # "paddle.nn.functional.leaky_relu": ["tensorlayerx.ops.leaky_relu", None],
    # # "paddle.nn.functional.pad": [" tensorlayerx.vision.transforms.pad", None],  # TODO - 用法不同
    "paddle.nn.functional.relu": ["tensorlayerx.ops.relu", None],
    # "paddle.nn.functional.gelu": ["tensorlayerx.ops.gelu", None],
    "paddle.nn.functional.sigmoid": ["tensorlayerx.ops.sigmoid", None],
    "paddle.nn.functional.softmax": ["tensorlayerx.ops.softmax", None],
    # "paddle.tanh": ["tensorlayerx.ops.tanh", None],
}


INIT_MAPPER = {
    # init
    # special case
    "paddle.fluid.param_attr.ParamAttr": ["tensorlayerx.nn.initializers.random_uniform", ParamAttrRandomMapper],  # TODO
    "paddle.ParamAttr": ["tensorlayerx.nn.initializers.xavier_uniform", ParamAttrXavierMapper],  # TODO
    # mapping
    "paddle.nn.initializer.Uniform": ["tensorlayerx.nn.initializers.random_uniform", None],  # TODO - 区别 - 参数默认值不同
    "paddle.nn.initializer.TruncatedNormal": ["tensorlayerx.nn.initializers.TruncatedNormal", TruncatedNormalOpMapper],
    "paddle.nn.initializer.KaimingNormal": ["tensorlayerx.nn.initializers.HeNormal", None],
    "paddle.nn.initializer.KaimingUniform": ["tensorlayerx.nn.initializers.HeUniform", None],
    "paddle.nn.initializer.XavierNormal": ["tensorlayerx.nn.initializers.XavierNormal", None],
    "paddle.nn.initializer.XavierUniform": ["tensorlayerx.nn.initializers.XavierUniform", None],  # TODO - 有点不一样
    "paddle.nn.initializer.Constant": ["tensorlayerx.nn.initializers.Constant", None],
    "paddle.ones": ["tensorlayerx.ones", None],
    "paddle.zeros": ["tensorlayerx.zeros", None],
}


LOSS_MAPPER = {
    # loss - nn
    "paddle.nn.CrossEntropyLoss": ["tensorlayerx.losses.softmax_cross_entropy_with_logits", None],  # TODO - 多分类
    "paddle.nn.BCELoss": ["tensorlayerx.losses.binary_cross_entropy", None],
    "paddle.nn.BCEWithLogitsLoss": ["tensorlayerx.losses.binary_cross_entropy", None],

    # loss - functional
    "paddle.nn.functional.binary_cross_entropy_with_logits": ["tensorlayerx.losses.binary_cross_entropy", None],
    "paddle.nn.functional.mse_loss": ["tensorlayerx.losses.mean_squared_error", None],
    "paddle.nn.functional.cosine_similarity": ["tensorlayerx.losses.cosine_similarity", None],
}


METRIC_MAPPER = {
    # metric
    "paddle.metric.accuracy": ["paddle.metric.accuracy", None],
    "paddle.mean": ["paddle.mean", None],
    "paddle.nn.functional.cross_entropy": ["paddle.nn.functional.cross_entropy", None]
}


OPTIMIZER_MAPPER = {
    "paddle.optimizer": ["tensorlayerx.optimizers", None],
    "paddle.optimizer.lr.ReduceOnPlateau": ["tensorlayerx.optimizers.lr.ReduceOnPlateau", None],
    "paddle.optimizer.lr.CosineAnnealingDecay": ["tensorlayerx.optimizers.lr.CosineAnnealingDecay", None],
    "paddle.optimizer.lr.MultiStepDecay": ["tensorlayerx.optimizers.lr.MultiStepDecay", None],
    "paddle.optimizer.Adam": ["tensorlayerx.optimizers.Adam", None],
    "paddle.optimizer.Momentum": ["tensorlayerx.optimizers.Momentum", None],
    "paddle.optimizer.SGD": ["tensorlayerx.optimizers.SGD", None],
}


UTILS_MAPPER = {
    "paddle.io.Dataset": ["tensorlayerx.dataflow.Dataset", None],
    "paddle.io.IterableDataset": ["tensorlayerx.dataflow.IterableDataset", None],
    "paddle.io.random_split": ["tensorlayerx.dataflow.random_split", None],
    # the follow api tlx is not support, so use paddle api
    "paddle.nn.functional.unfold": ["paddle.nn.functional.unfold", None],
    "paddle.floor": ["paddle.floor", None],
    "paddle.shape": ["paddle.shape", None],
    "paddle.rand": ["paddle.rand", None],

}


VISION_MAPPER = {
    "paddle.vision": ["tensorlayerx.vision", None],
    # data enhancement & utils
    "paddle.vision.transforms": ["tensorlayerx.vision.transforms", None],
    "paddle.vision.transforms.adjust_brightness": ["tensorlayerx.vision.transforms.AdjustBrightness", None],
    "paddle.vision.transforms.adjust_contrast": ["tensorlayerx.vision.transforms.AdjustContrast", None],
    "paddle.vision.transforms.adjust_hue": ["tensorlayerx.vision.transforms.AdjustHue", None],
    "class paddle.vision.transforms.ColorJitter": ["tensorlayerx.vision.transforms.ColorJitter", None],
    "paddle.vision.transforms.Compose": ["tensorlayerx.vision.transforms.Compose", None],
    "paddle.vision.transforms.crop": ["tensorlayerx.vision.transforms.Crop", None],
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
    "paddle.vision.transforms.rotate": ["tensorlayerx.vision.transforms.Rotation", None],  # TODO - 用法有区别
    "paddle.vision.transforms.SaturationTransform": ["tensorlayerx.vision.transforms.AdjustSaturation", None],
    "paddle.vision.transforms.to_grayscale": ["tensorlayerx.vision.transforms.RgbToGray", None],  # TODO - 用法有区别
    "paddle.vision.transforms.ToTensor": ["tensorlayerx.vision.transforms.ToTensor", None],  # CHW -> HWC
    "paddle.vision.transforms.Transpose": ["tensorlayerx.vision.transforms.Transpose", None],
    "paddle.vision.transforms.vflip": ["tensorlayerx.vision.transforms.FlipVertical", None],

    # load
    "paddle.vision.image.image_load": ["tensorlayerx.vision.utils.load_image", None],  # TODO - 有区别
    # "": ["tensorlayerx.vision.utils.load_images", None],
    # "": ["tensorlayerx.vision.utils.save_images", None],
    # metric
    # "": ["tensorlayerx.vision.nms", None],
    # "": ["tensorlayerx.vision.box_iou", None],
    # "": ["tensorlayerx.vision.box_area", None],
}


API_MAPPER = {
    "paddle": ["tensorlayerx", None],
    "paddle.load": ["paddle.load", None],  # TODO
    # "paddle.load": ["tensorlayerx.files.load_npz", None],  # load_npz - load_and_assign_npz_dict
    # "paddle.save": ["tensorlayerx.files.save_npz", None],  # save_npz - save_npz_dict
    "paddle.set_device": ["tensorlayerx.set_device", None],
    "paddle.device.get_device": ["tensorlayerx.get_device", None],
    # "paddle.": ["tensorlayerx.to_device", None],
    "paddle.concat": ["tensorlayerx.ops.concat", None],
    "paddle.to_tensor": ["tensorlayerx.ops.convert_to_tensor", None],
    "paddle.seed": ["tensorlayerx.set_seed", None],
    "paddle.unsqueeze": ["tensorlayerx.ops.expand_dims", None],
    "paddle.squeeze": ["tensorlayerx.ops.squeeze", None],
    "paddle.transpose": ["tlx.ops.transpose", None],
    "paddle.reshape": ["tlx.ops.reshape", None],
    # "paddle.sum": ["", None],
    # "paddle.mean": ["", None],
    "paddle.ones": ["tensorlayerx.ops.ones", None],
    "paddle.zeros": ["tensorlayerx.ops.zeros", None],
    "paddle.sqrt": ["tensorlayerx.ops.sqrt", None],
    "paddle.arange": ["tensorlayerx.ops.arange", None],
    "paddle.matmul": ["tensorlayerx.ops.matmul", None],
    # "paddle.clip": ["", None],
    "paddle.exp": ["tensorlayerx.ops.exp", None],
    # "paddle.max": ["", None],
    # "paddle.min": ["", None],
    "paddle.argmax": ["tensorlayerx.ops.argmax", None],
    "paddle.argmin": ["tensorlayerx.ops.argmin", None],
    "paddle.stack": ["tensorlayerx.ops.stack", None],
    "paddle.log": ["tensorlayerx.ops.log", None],
    "paddle.abs": ["tensorlayerx.ops.abs", None],
    "paddle.logical_or": ["tensorlayerx.ops.logical_or", None],
    "paddle.logical_xor": ["tensorlayerx.ops.logical_xor", None],
    "paddle.logical_and": ["tensorlayerx.ops.logical_and", None],
    "paddle.logical_not": ["tensorlayerx.ops.logical_not", None],
    "paddle.split": ["tensorlayerx.ops.split", SplitOpMapper],  # TODO - BUG在0.5.7已修复
    "paddle.add": ["paddle.add", None],  # tensorlayerx.ops.add - not support
    "paddle.multiply": ["tensorlayerx.ops.multiply", None],
    "paddle.einsum": ["tensorlayerx.ops.einsum", None],
}

CUSTOM_API = {
    # "model.load_dict": ["restore_model", LoadModelTLXMapper],  # deprecated
    "load_model": ["restore_model", LoadModelTLXMapper],
    "_load_pretrained": ["restore_model", LoadModelTLXMapper],
    # ops
    "_norm_layer": ["_norm_layer", BatchNormModuleMapper],
    "norm_layer": ["norm_layer", BatchNormModuleMapper],  # 为了区分建议改成 batch_norm
    "layer_norm": ["layer_norm", LayerNormModuleMapper],
    # "self._norm_layer": ["self._norm_layer", BatchNormModuleMapper],
    "partial": ["partial", PartialTLXMapper],
    # "act_layer": ["act_layer", CustomFuncActMapper],
}

# CUSTOM_ASSIGN_MAP = {
#     "self.bn1": "self._norm_layer"
# }


INVALID_API = {
    "paddle.utils.download.get_weights_path_from_url": ["paddle.utils.download.get_weights_path_from_url", None],
    # "paddle.nn.functional": ["tensorlayerx.ops", None],
}


API_MAPPER.update(NN_MAPPER)
API_MAPPER.update(INIT_MAPPER)
API_MAPPER.update(LOSS_MAPPER)
API_MAPPER.update(METRIC_MAPPER)
API_MAPPER.update(OPTIMIZER_MAPPER)
API_MAPPER.update(UTILS_MAPPER)
API_MAPPER.update(VISION_MAPPER)
API_MAPPER.update(CUSTOM_API)
API_MAPPER.update(INVALID_API)


# when this op as parameter assignment
REPLACE_API = {
    "ParamAttr(initializer=Uniform(-stdv, stdv))": "random_uniform(-stdv, stdv)",  # alexnet
    "ParamAttr": "tensorlayerx.nn.initializers.xavier_uniform",  # regnet
    "BatchNorm2D": "BatchNorm2d",
    "nn.BatchNorm2D": "nn.BatchNorm2d",  # 需要适配
    "nn.Hardsigmoid": "nn.HardSigmoid",
    "nn.GELU": "tensorlayerx.ops.gelu",
}

HEADER_IMPORT = {
    "__future__",
}

NOT_SUPPORT_API = {

}

FUNC2VARIABLE = {
    "act_layer()": "act_layer"
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
    # TODO - tlx无对应api支持
    "paddle.fluid.param_attr.ParamAttr",
    "paddle.nn.functional",  # not import
    # custom api
    "load_model.load_model",
    "load_model._load_pretrained"
]
