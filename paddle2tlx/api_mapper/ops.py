# coding: utf-8
from .utils import *


# class LoadModelTLXMapper(Mapper):
#     def __init__(self,
#                  func_name,
#                  paddle_api_name,
#                  args,
#                  kwargs,
#                  target_name=None):
#         super().__init__(func_name, paddle_api_name, args, kwargs, target_name)
#
#     def process_attrs(self):
#         self.kwargs["tlx_model"] = "model"  # TODO
#
#     def run(self):
#         if self.paddle_api_name == "model.load_dict" and self.rename_func_name(
#                 "restore_model"):
#             return [], generate_api_code(self.func_name, self.args,
#                                          self.kwargs), []
#         else:
#             self.convert_args2kwargs(1)
#             return self.convert_to_tlx()


class LoadModelTLXMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        old_args = self.args
        self.args = []
        for name in old_args:
            if name.lower() == "model_urls":
                continue
            self.args.append(name)
        # load_direct: set False in debug mode, set True when test converted project
        self.kwargs["load_direct"] = False

    def run(self):
        if self.paddle_api_name == "load_model" and self.rename_func_name("load_model"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class PartialTLXMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        if len(self.args) == 1:
            param_val = self.args[0]
            if param_val == "nn.BatchNorm2D":
                param_val = "nn.BatchNorm2d"
            self.args = []
            self.args.append(param_val)

    def run(self):
        if self.paddle_api_name == "partial" and self.rename_func_name("partial"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class LoadImageMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        if "mode" in self.kwargs:
            self.kwargs["mode"] = "tlx" if self.kwargs["mode"] == "pd" else "tlx"

    def run(self):
        return self.convert_to_tlx()


class SplitOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "num_or_sections", "num_or_size_splits")
        # pass

    def run(self):
        if self.paddle_api_name == "paddle.split" and self.rename_func_name("tensorlayerx.split"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class AddMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, 'x', 'value')
        rename_key(self.kwargs, 'y', 'bias')
        # 待添加:去除names

    def run(self):
        return self.convert_to_tlx()


class ClipOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "x", "t")
        rename_key(self.kwargs, "min", "clip_value_min")
        rename_key(self.kwargs, "max", "clip_value_max")
        if not "clip_value_min" in self.kwargs:
            self.kwargs["clip_value_min"] = None
        if not "clip_value_max" in self.kwargs:
            self.kwargs["clip_value_max"] = None

    def run(self, paddle2tlx_func_name=None):
        if self.paddle_api_name == "paddle.clip" and self.rename_func_name("tensorlayerx.clip_by_value"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            return self.convert_to_tlx()


class RollOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "axis", "dims")

    def run(self, paddle2tlx_func_name=None):
        if self.paddle_api_name == "paddle.roll" and self.rename_func_name("tensorlayerx.roll"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            return self.convert_to_tlx()


class ArgmaxOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        delete_key(self.kwargs, "keepdim")

    def run(self):
        return self.convert_to_tlx()


class StackOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        rename_key(self.kwargs, "x", "values")
        if "name" in self.kwargs:
            delete_key(self.kwargs, "name")

    def run(self):
        return self.convert_to_tlx()


class FullOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "fill_value", "value")

    def run(self):
        return self.convert_to_tlx()


class FullLikeOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        if len(self.args) == 2:
            self.kwargs["shape"] = self.args[0] + ".shape"
            self.kwargs["value"] = self.args[1]
            self.args = []  # clear
        # rename_key(self.kwargs, "fill_value", "value")
        # rename_key(self.kwargs, "x", "shape")
        # delete_key(self.kwargs, "name")
        # if "shape" in self.kwargs:
        #     self.kwargs["shape"] = 'tlx.get_tensor_shape(' + self.kwargs["shape"] + ')'

    def run(self):
        return self.convert_to_tlx()


class FlattenOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        if len(self.args) == 0:
            self.args = []

    def delete_attrs(self):
        # delete_key(self.kwargs, "start_axis")
        # delete_key(self.kwargs, "stop_axis")
        pass

    def run(self):
        if self.paddle_api_name == "paddle.flatten" and self.rename_func_name("tensorlayerx.flatten"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class SliceOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "ends", "sizes")

    def run(self):
        if self.paddle_api_name == "paddle.slice" and self.rename_func_name("tensorlayerx.slice"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class AdaptionAvgPool2dOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "x", "input")

    def run(self):
        if self.paddle_api_name == "paddle.nn.functional.adaptive_avg_pool2d" and self.rename_func_name("tensorlayerx.ops.adaptive_avg_pool2d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class AdaptionMaxPool2dOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "x", "input")

    def run(self):
        if self.paddle_api_name == "paddle.nn.functional.adaptive_max_pool2d" and self.rename_func_name("tensorlayerx.ops.adaptive_max_pool2d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class ExpandOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        delete_key(self.kwargs, "name")

    def run(self):
        if self.paddle_api_name == "paddle.expand" and self.rename_func_name("tensorlayerx.expand"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class TopkOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "axis", "dim")

    def run(self):
        if self.paddle_api_name == "paddle.topk" and self.rename_func_name("tensorlayerx.topk"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class GatherOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "index", "indices")

    def run(self):
        if self.paddle_api_name == "paddle.gather" and self.rename_func_name("tensorlayerx.gather"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class DataLoaderMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        delete_key(self.kwargs, "return_list")
        delete_key(self.kwargs, "use_shared_memory")

    def run(self):
        return self.convert_to_tlx()


class DistributedBatchSamplerMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        delete_key(self.kwargs, "shuffle")

    def run(self):
        return self.convert_to_tlx()


class Pad2DModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        self.kwargs["data_format"] = "channels_first"
        if self.kwargs["data_format"] == "NHWC":
            self.kwargs["data_format"] = "channels_first"
        # print("before pad2d.padding", self.kwargs["padding"])

    def check_attrs(self):
        """ 确认参数的值。
        """
        pass
        # import json
        # self.kwargs["padding"] = json.loads(self.kwargs["padding"])
        # if "padding" not in self.kwargs:
        #     pass
        # else:
        #     if isinstance(self.kwargs["padding"], int):
        #         pass
        #     elif any(key in self.kwargs["padding"] for key in ['//']):
        #         pass
        #     elif isinstance(self.kwargs["padding"], (tuple,list)):
        #         try:
        #             self.kwargs["padding"] = [int(ele.strip()) for ele in self.kwargs["padding"][1:-2].split(',')]
        #         except:
        #             self.kwargs["padding"] = [ele.strip() for ele in self.kwargs["padding"][1:-2].split(',')]
        #         self.kwargs["padding"] = ((self.kwargs["padding"][0],self.kwargs["padding"][1]),
        #                                   (self.kwargs["padding"][2],self.kwargs["padding"][3]))

    def run(self):
        return self.convert_to_tlx()


class MeanOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "keepdim", "keepdims")

    def run(self):
        return self.convert_to_tlx()


class ReshapeMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "x", "tensor")
        delete_key(self.kwargs, 'name')

    def run(self):
        return self.convert_to_tlx()


class SumOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "keepdim", "keepdims")
        delete_key(self.kwargs, "name")

    def run(self):
        return self.convert_to_tlx()


class CumsumOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        # pd:(x, axis=None,dtype=None,name=None)
        delete_key(self.kwargs, "name")

    def run(self):
        return self.convert_to_tlx()


class ConcatOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "x", "values")
        delete_key(self.kwargs, "name")

    def run(self):
        return self.convert_to_tlx()


class MaxOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "keepdim", "keepdims")

    def run(self):
        return self.convert_to_tlx()


class MatMulOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "transpose_x", "transpose_a")
        rename_key(self.kwargs, "transpose_y", "transpose_b")

    def run(self):
        return self.convert_to_tlx()


class BatchSamplerModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        delete_key(self.kwargs, "shuffle")

    def run(self):
        return self.convert_to_tlx()


class BCEwithLogitLossMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        # paddle:(logit, label, weight=None, reduction='mean', pos_weight=None, name=None)
        # tensorlayerx:sigmoid_cross_entropy(output, target, reduction='mean')
        rename_key(self.kwargs, "logit", "output")
        rename_key(self.kwargs, "label", "target")

    def run(self):
        if self.paddle_api_name == "paddle.nn.functional.binary_cross_entropy_with_logits" and \
                self.rename_func_name("tensorlayerx.ops.sigmoid_cross_entropy"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class SigmoidCrossEntropyOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        delete_key(self.kwargs, 'alpha')
        delete_key(self.kwargs, 'gamma')

    def run(self):
        if self.paddle_api_name == "paddle.nn.functional.sigmoid_focal_loss" \
                and self.rename_func_name("tensorlayerx.losses.sigmoid_cross_entropy"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class ArangeOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "step", "delta")
        rename_key(self.kwargs, "end", "limit")

    def run(self):
        if self.paddle_api_name == "paddle.arange" and self.rename_func_name("tensorlayerx.arange"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class AddOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        if len(self.kwargs) == 0:
            if "name" in self.args:
                self.args.remove("name")
        else:
            rename_key(self.kwargs, "x", "value")
            rename_key(self.kwargs, "y", "bias")

    def run(self):
        return self.convert_to_tlx()
