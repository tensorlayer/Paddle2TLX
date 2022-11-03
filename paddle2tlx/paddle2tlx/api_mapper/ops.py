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
#             return self.convert_to_paddle()


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
        # self.args.append("**kwargs")
        # self.args.append("load_direct")
        self.kwargs["load_direct"] = False  # set True when test converted project

    def run(self):
        if self.paddle_api_name == "load_model" and self.rename_func_name("load_model"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_paddle()


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
            return self.convert_to_paddle()


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

    def run(self):
        if self.paddle_api_name == "paddle.split" and self.rename_func_name("tensorlayerx.ops.split"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_paddle()


class TruncatedNormalOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "std", "stddev")

    def run(self):
        if self.paddle_api_name == "paddle.nn.initializer.TruncatedNormal" and \
                self.rename_func_name("tensorlayerx.nn.initializers.TruncatedNormal"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(1)
            return self.convert_to_paddle()


class CustomFuncActMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        pass

    def run(self):
        return self.convert_to_paddle()
