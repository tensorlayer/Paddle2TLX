# coding: utf-8
from .utils import *


class ConvModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "weight_attr", "W_init")
        rename_key(self.kwargs, "bias_attr", "b_init")
        delete_key(self.kwargs, "groups")
        delete_key(self.kwargs, "dilation")
        if "padding" not in self.kwargs:
            self.kwargs["padding"] = 0  # default value is different
        self.kwargs["data_format"] = "channels_first"

    def run(self):
        if self.paddle_api_name == "paddle.nn.Conv1D" and self.rename_func_name(
                "tensorlayerx.nn.Conv1d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.Conv2D" and self.rename_func_name(
                "tensorlayerx.nn.Conv2d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.Conv3D" and self.rename_func_name(
                "tensorlayerx.nn.Conv3d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(0)  # 7
            return self.convert_to_paddle()


class GroupConvModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "weight_attr", "W_init")
        rename_key(self.kwargs, "bias_attr", "b_init")
        rename_key(self.kwargs, "groups", "n_group")
        if "padding" not in self.kwargs:
            self.kwargs["padding"] = 0  # default value is different
        # if "W_init" in self.kwargs:  # need optimize
        #     if "initializer" not in self.kwargs["W_init"]:
        #         start = self.kwargs["W_init"].find("(")
        #         param_val_new = self.kwargs["W_init"][0:start] + "()"
        #         self.kwargs["W_init"] = param_val_new
        # if "b_init" in self.kwargs:
        #     if "initializer" not in self.kwargs["b_init"]:
        #         start = self.kwargs["b_init"].find("(")
        #         param_val_new = self.kwargs["b_init"][0:start] + "()"
        #         self.kwargs["b_init"] = param_val_new
        self.kwargs["data_format"] = "channels_first"

    def run(self):
        if self.paddle_api_name == "paddle.nn.Conv1D" and self.rename_func_name(
                "tensorlayerx.nn.GroupConv1d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.Conv2D" and self.rename_func_name(
                "tensorlayerx.nn.GroupConv2d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.Conv3D" and self.rename_func_name(
                "tensorlayerx.nn.GroupConv3d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(0)
            return self.convert_to_paddle()


class MaxPoolModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        if "padding" not in self.kwargs:
            self.kwargs["padding"] = 0
        self.kwargs["data_format"] = "channels_first"

    def check_attrs(self):
        assert "return_mask" not in self.kwargs, "The return_mask is not supported yet in MaxPool!"

    def run(self):
        if self.paddle_api_name == "paddle.nn.MaxPool1d" and self.rename_func_name(
                "tensorlayerx.MaxPool1D"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.MaxPool2d" and self.rename_func_name(
                "tensorlayerx.MaxPool12D"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.MaxPool3d" and self.rename_func_name(
                "tensorlayerx.MaxPool13D"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(3)
            return self.convert_to_paddle()


class AvgPoolFuncMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        if "padding" not in self.kwargs:
            self.kwargs["padding"] = 0
        self.kwargs["data_format"] = "channels_first"
        delete_key(self.kwargs, "exclusive")

    def run(self):
        if self.paddle_api_name == "paddle.nn.AvgPool1D" and self.rename_func_name(
                "tensorlayerx.nn.AvgPool1d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.AvgPool2D" and self.rename_func_name(
                "tensorlayerx.nn.AvgPool2d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.AvgPool3D" and self.rename_func_name(
                "tensorlayerx.nn.AvgPool3d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(3)
            return self.convert_to_paddle()


class AdaptiveAvgPoolMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        # modify parameter default value, data_format 'NCHW' -> 'channels_first'
        self.kwargs["data_format"] = "channels_first"

    def run(self):
        if self.paddle_api_name == "paddle.nn.AdaptiveAvgPool1D" and self.rename_func_name(
                "tensorlayerx.nn.AdaptiveAvgPool1d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.AdaptiveAvgPool2D" and self.rename_func_name(
                "tensorlayerx.nn.AdaptiveAvgPool2d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.AdaptiveAvgPool3D" and self.rename_func_name(
                "tensorlayerx.nn.AdaptiveAvgPool3d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(3)
            return self.convert_to_paddle()


class LinearModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "weight_attr", "W_init")
        rename_key(self.kwargs, "bias_attr", "b_init")
        if "W_init" in self.kwargs:
            if "initializer" in self.kwargs["W_init"]:
                self.kwargs["W_init"] = "tensorlayerx.initializers.random_uniform(-stdv, stdv)"
        if "b_init" in self.kwargs:
            if "initializer" not in self.kwargs["b_init"]:
                self.kwargs["b_init"] = "tensorlayerx.initializers.xavier_uniform()"

    def run(self):
        if self.paddle_api_name == "paddle.nn.Linear" and self.rename_func_name("tensorlayerx.nn.Linear"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(0)
            return self.convert_to_paddle()


class FlattenModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        delete_key(self.kwargs, "start_axis")
        delete_key(self.kwargs, "stop_axis")

    def run(self):
        if self.paddle_api_name == "paddle.flatten" and self.rename_func_name("tensorlayerx.nn.Flatten()"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(1)
            return self.convert_to_paddle()


class DropoutModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        delete_key(self.kwargs, "mode")

    def run(self):
        if self.paddle_api_name == "paddle.nn.functional.dropout" and self.rename_func_name("nn.Dropout"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(1)
            return self.convert_to_paddle()


class BatchNormModuleMapper(Mapper):
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
            self.args = []
            self.kwargs["num_features"] = param_val
        delete_key(self.kwargs, "param_attr")
        delete_key(self.kwargs, "bias_attr")
        rename_key(self.kwargs, "moving_mean_name", "moving_mean_init")
        rename_key(self.kwargs, "moving_variance_name", "moving_var_init")
        if "moving_mean_init" in self.kwargs:
            if "initializer" not in self.kwargs["moving_mean_init"]:
                self.kwargs["moving_mean_init"] = "tensorlayerx.initializers.xavier_uniform()"
        if "moving_var_init" in self.kwargs:
            if "initializer" not in self.kwargs["moving_var_init"]:
                self.kwargs["moving_var_init"] = "tensorlayerx.initializers.xavier_uniform()"
        self.kwargs["data_format"] = "channels_first"

    def run(self):
        if self.paddle_api_name == "paddle.nn.BatchNorm" and self.rename_func_name("tensorlayerx.nn.BatchNorm"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        if self.paddle_api_name == "paddle.nn.BatchNorm1D" and self.rename_func_name("tensorlayerx.nn.BatchNorm1d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.BatchNorm1D" and self.rename_func_name("tensorlayerx.nn.BatchNorm2d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.BatchNorm3D" and self.rename_func_name("tensorlayerx.nn.BatchNorm3d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(0)
            return self.convert_to_paddle()


class LayerNormModuleMapper(Mapper):
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
        if self.paddle_api_name == "paddle.nn.LayerNorm" and self.rename_func_name("tensorlayerx.nn.LayerNorm"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(0)
            return self.convert_to_paddle()
