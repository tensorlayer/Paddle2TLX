# coding: utf-8
import select

from .utils import *


class SequentialModuleMapper(Mapper):
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
        if self.paddle_api_name == "paddle.nn.Sequential" and self.rename_func_name("tensorlayerx.nn.Sequential"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(0)
            return self.convert_to_tlx()


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
        if "padding" not in self.kwargs:
            self.kwargs["padding"] = 0  # default value is different
        self.kwargs["data_format"] = "channels_first"

    def delete_attrs(self):
        delete_key(self.kwargs, "groups")
        delete_key(self.kwargs, "dilation")

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
            return self.convert_to_tlx()


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
        else:
            padding_val = self.kwargs["padding"]
            if isinstance(padding_val, int):
                pass
            elif any(key in padding_val for key in ['//']):
                pass
            elif padding_val.isalpha():
                pass
            elif not any((key in padding_val for key in ['+', '-', '*', '/'])) and "_" not in padding_val:
                self.kwargs["padding"] = "tuple({})".format(padding_val)
        if "W_init" in self.kwargs:  # need optimize
            if self.kwargs["W_init"].lower() == "true":
                self.kwargs["W_init"] = "tensorlayerx.nn.initializers.constant"
            # elif self.kwargs["b_init"].split('\n')[0] == "False":
            elif self.kwargs["W_init"].lower() == "false":
                self.kwargs["W_init"] = "None"
            elif "initializer" not in self.kwargs["W_init"] and self.kwargs["W_init"].strip() != "None" \
                    and self.kwargs["W_init"].strip() != "W_init":  # seg
                start = self.kwargs["W_init"].find("(")
                if start != -1:
                    param_val_new = self.kwargs["W_init"][0:start] + "()"
                    self.kwargs["W_init"] = param_val_new
            if "HeNormal" in self.kwargs["W_init"]:
                delete_key(self.kwargs, "W_init")
            elif "L2Decay" in self.kwargs["W_init"]:
                delete_key(self.kwargs, "W_init")
        if "b_init" in self.kwargs:
            if self.kwargs["b_init"].lower() == "true":
                self.kwargs["b_init"] = "tensorlayerx.nn.initializers.constant"
            # elif self.kwargs["b_init"].split('\n')[0] == "False":
            elif self.kwargs["b_init"].lower() == "false":
                self.kwargs["b_init"] = "None"
            elif "initializer" not in self.kwargs["b_init"] and self.kwargs["b_init"].strip() != "None" \
                    and self.kwargs["b_init"] != "b_init":  # seg
                start = self.kwargs["b_init"].find("(")
                if start != -1:
                    param_val_new = self.kwargs["b_init"][0:start] + "()"
                    self.kwargs["b_init"] = param_val_new
            if self.kwargs["b_init"].strip() == 'True':
                delete_key(self.kwargs, "b_init")
            elif "L2Decay" in self.kwargs["b_init"]:
                delete_key(self.kwargs, "b_init")
        if "data_format" not in self.kwargs:
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
            return self.convert_to_tlx()


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

    def delete_attrs(self):
        delete_key(self.kwargs, "return_mask")

    # def check_attrs(self):
    #     assert "return_mask" not in self.kwargs, "The return_mask is not supported yet in MaxPool!"

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
            return self.convert_to_tlx()


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

    def delete_attrs(self):
        delete_key(self.kwargs, "exclusive")
        delete_key(self.kwargs, "ceil_mode")

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
            return self.convert_to_tlx()


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
            return self.convert_to_tlx()


class AdaptiveMaxPoolMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        self.kwargs["data_format"] = "channels_first"

    def delete_attrs(self):
        delete_key(self.kwargs, "return_mask")

    def run(self):
        if self.paddle_api_name == "paddle.nn.AdaptiveMaxPool1D" and self.rename_func_name(
                "tensorlayerx.nn.AdaptiveMaxPool1d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.AdaptiveMaxPool2D" and self.rename_func_name(
                "tensorlayerx.nn.AdaptiveMaxPool2d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.AdaptiveMaxPool3D" and self.rename_func_name(
                "tensorlayerx.nn.AdaptiveMaxPool3d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(1)
            return self.convert_to_tlx()


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
            if "initializer" in self.kwargs["W_init"] and "(" in self.kwargs["W_init"]:
                self.kwargs["W_init"] = "tensorlayerx.initializers.random_uniform(-stdv, stdv)"
            if "name=" in self.kwargs["W_init"]:
                delete_key(self.kwargs, "W_init")
        if "b_init" in self.kwargs:
            if "initializer" not in self.kwargs["b_init"] and "(" in self.kwargs["b_init"]:
                self.kwargs["b_init"] = "tensorlayerx.initializers.xavier_uniform()"
            if "name=" in self.kwargs["b_init"]:
                delete_key(self.kwargs, "b_init")
            # # TODO - 运行时确定, 已修改成事先在代码中适配
            # if "qkv_bias" in self.kwargs["b_init"] or "bias" in self.kwargs["b_init"]:
            #     delete_key(self.kwargs, "b_init")

    def run(self):
        if self.paddle_api_name == "paddle.nn.Linear" and self.rename_func_name("tensorlayerx.nn.Linear"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(0)
            return self.convert_to_tlx()


class LinearOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        # rename_key(self.kwargs, "x", "a")
        # rename_key(self.kwargs, "weight", "b")
        # paddle.nn.functional.linear(x, weight, bias=None, name=None)
        # tensorlayerx.ops.linear(input, weight, bias = None)
        rename_key(self.kwargs, "x", "input")

    def run(self):
        # if self.paddle_api_name == "paddle.nn.functional.linear" and self.rename_func_name("tensorlayerx.ops.matmul"):
        if self.paddle_api_name == "paddle.nn.functional.linear" and self.rename_func_name("tensorlayerx.ops.linear"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            # self.convert_args2kwargs(0)
            return self.convert_to_tlx()


class L1LossOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "label", "target")
        delete_key(self.kwargs, "name")

    def run(self):
        if self.paddle_api_name == "paddle.nn.functional.l1_loss" and self.rename_func_name("tensorlayerx.losses.L1Loss"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            # self.convert_args2kwargs(0)
            return self.convert_to_tlx()


class FlattenModuleMapper(Mapper):
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
        delete_key(self.kwargs, "start_axis")
        delete_key(self.kwargs, "stop_axis")

    def run(self):
        # if self.paddle_api_name == "paddle.flatten" and self.rename_func_name("tensorlayerx.flatten"):
        if self.paddle_api_name == "paddle.nn.Flatten" and self.rename_func_name("tensorlayerx.nn.Flatten"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(1)
            return self.convert_to_tlx()


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
            return self.convert_to_tlx()


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
            if "norm_type" in self.kwargs or "norm_decay" in self.kwargs:  # todo - det
                return
            self.args = []
            self.kwargs["num_features"] = param_val
        rename_key(self.kwargs, "moving_mean_name", "moving_mean_init")
        rename_key(self.kwargs, "moving_variance_name", "moving_var_init")
        rename_key(self.kwargs, "num_channels", "num_features")
        if "moving_mean_init" in self.kwargs:
            # if "L2Decay" in self.kwargs["moving_mean_init"]:
            #     delete_key(self.kwargs, "moving_mean_init")
            if "initializer" not in self.kwargs["moving_mean_init"]:
                self.kwargs["moving_mean_init"] = "tensorlayerx.initializers.xavier_uniform()"
        if "moving_var_init" in self.kwargs:
            if "initializer" not in self.kwargs["moving_var_init"]:
                self.kwargs["moving_var_init"] = "tensorlayerx.initializers.xavier_uniform()"
        self.kwargs["data_format"] = "channels_first"

    def delete_attrs(self):
        delete_key(self.kwargs, "param_attr")
        delete_key(self.kwargs, "weight_attr")  # det
        delete_key(self.kwargs, "bias_attr")
        delete_key(self.kwargs, "data_layout")
        delete_key(self.kwargs, "use_global_stats")

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
            return self.convert_to_tlx()


class LayerNormModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "weight_attr", "gamma_init")
        rename_key(self.kwargs, "bias_attr", "beta_init")

        if "gamma_init" in self.kwargs:
            if self.kwargs["gamma_init"].lower() == "true":
                self.kwargs["gamma_init"] = "tensorlayerx.nn.initializers.constant"
            # elif self.kwargs["b_init"].split('\n')[0] == "False":
            elif self.kwargs["gamma_init"].lower() == "fasle":
                self.kwargs["gamma_init"] = "None"

        if "beta_init" in self.kwargs:
            if self.kwargs["beta_init"].lower() == "true":
                self.kwargs["beta_init"] = "tensorlayerx.nn.initializers.zeros"
            elif self.kwargs["beta_init"].lower() == "fasle":
                self.kwargs["beta_init"] = "None"

        # if "data_format" not in self.kwargs:  # log-20230103
        #     self.kwargs["data_format"] = "channels_first"

    def run(self):
        if self.paddle_api_name == "paddle.nn.LayerNorm" and self.rename_func_name("tensorlayerx.nn.LayerNorm"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(0)
            return self.convert_to_tlx()


class InstanceNormModuleMapper(Mapper):
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
        rename_key(self.kwargs, "weight_attr", "gamma_init")
        rename_key(self.kwargs, "bias_attr", "beta_init")
        self.kwargs["data_format"] = "channels_first"

    def delete_attrs(self):
        delete_key(self.kwargs, "param_attr")
        delete_key(self.kwargs, "bias_attr")
        delete_key(self.kwargs, "data_layout")

    def check_attrs(self):
        """ 确认参数的值。
        """
        # print(f"instancenorm=",self.kwargs)
        if "beta_init" in self.kwargs.keys():
            if self.kwargs["beta_init"].split('\n')[0] == "True":
                self.kwargs["beta_init"] = "tensorlayerx.nn.initializers.Constant(0.0)"
            elif self.kwargs["beta_init"].split('\n')[0] == "False":
                self.kwargs["beta_init"] = "None"
        if "gamma_init" in self.kwargs.keys():
            if self.kwargs["gamma_init"].split('\n')[0] == "True":
                self.kwargs["gamma_init"] = "tensorlayerx.nn.initializers.Constant(1.0)"
            elif self.kwargs["gamma_init"].split('\n')[0] == "False":
                self.kwargs["gamma_init"] = "None"
            # if self.kwargs["gamma_init"]

    def run(self):
        if self.paddle_api_name == "paddle.nn.InstanceNorm" and self.rename_func_name("tensorlayerx.nn.InstanceNorm"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        if self.paddle_api_name == "paddle.nn.InstanceNorm1D" and self.rename_func_name("tensorlayerx.nn.InstanceNorm1d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.InstanceNorm2D" and self.rename_func_name("tensorlayerx.nn.InstanceNorm2d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.paddle_api_name == "paddle.nn.InstanceNorm3D" and self.rename_func_name("tensorlayerx.nn.InstanceNorm3d"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(0)
            return self.convert_to_tlx()


class DeformConv2DMapper(Mapper):
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
            if "learning_rate=" in self.kwargs["W_init"]:
                delete_key(self.kwargs, "W_init")
        if "b_init" in self.kwargs:
            if 'False' in self.kwargs['b_init']:
                self.kwargs['b_init'] = 'None'

    def run(self):
        return self.convert_to_tlx()


class DeformableConvV2Mapper(Mapper):
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
        self.kwargs['data_format'] = 'channels_first'

    def run(self):
        return self.convert_to_tlx()


class DeformableConvV2NewMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        delete_key(self.kwargs, "weight_attr")
        delete_key(self.kwargs, "bias_attr")
        delete_key(self.kwargs, "regularizer")

    def run(self):
        return self.convert_to_tlx()


class UpsampleModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        self.kwargs['data_format'] = 'channels_first'  # picodet_lcnet

    def run(self):
        return self.convert_to_tlx()


class MaxUnPool2DModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        self.kwargs['data_format'] = 'channels_first'  # bit

    def run(self):
        return self.convert_to_tlx()


class Conv2DTransposeMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        if len(self.args) == 2:
            self.kwargs['in_channels'] = self.args[0]
            self.kwargs['out_channels'] = self.args[1]
        self.kwargs['data_format'] = 'channels_first'
        if 'bias_attr' in self.kwargs.keys():
            self.kwargs["b_init"] = self.kwargs["bias_attr"]
        if "weight_attr" in self.kwargs.keys():
            self.kwargs["W_init"] = self.kwargs["weight_attr"]
        rename_key(self.kwargs, 'groups', 'n_group')
        if "padding" not in self.kwargs:
            self.kwargs["padding"] = 0

    def check_attrs(self):
        # return super().check_attrs()
        if "b_init" in self.kwargs.keys():
            if self.kwargs["b_init"].split('\n')[0] == "True":
                self.kwargs["b_init"] = "tensorlayerx.nn.initializers.constant"
            elif self.kwargs["b_init"].split('\n')[0] == "False":
                self.kwargs["b_init"] = "None"
        if "W_init" in self.kwargs.keys():
            if self.kwargs["W_init"].split('\n')[0] == "True":
                self.kwargs["W_init"] = "tensorlayerx.nn.initializers.truncated_normal"
            elif self.kwargs["W_init"].split('\n')[0] == "False":
                self.kwargs["W_init"] = "None"

    def delete_attrs(self):
        delete_key(self.kwargs, "bias_attr")
        delete_key(self.kwargs, "weight_attr")

    def run(self):
        return self.convert_to_tlx()


class UpSamplingModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "scale_factor", "scale")
        rename_key(self.kwargs, "mode", "method")
        self.kwargs["data_format"] = "channels_first"

    def run(self):
        return self.convert_to_tlx()


class MultiHeadAttentionMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        self.kwargs['bias'] = True
        self.kwargs['batch_first'] = True

    def delete_attrs(self):
        delete_key(self.kwargs, "weight_attr")
        delete_key(self.kwargs, "bias_attr")

    def run(self):
        return self.convert_to_tlx()


class RnnMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        self.kwargs['bias'] = True
        self.kwargs['batch_first'] = True
        rename_key(self.kwargs, "activation", "act")

    def delete_attrs(self):
        delete_key(self.kwargs, "weight_ih_attr")
        delete_key(self.kwargs, "bias__ih_attr")
        delete_key(self.kwargs, "weight_hh_attr")
        delete_key(self.kwargs, "bias_hh_attr")
        delete_key(self.kwargs, "direction")
        delete_key(self.kwargs, "time_major")

    def run(self):
        return self.convert_to_tlx()


class LstmMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        self.kwargs['bias'] = True
        self.kwargs['batch_first'] = True
        self.kwargs['bidirectional'] = False

        rename_key(self.kwargs, "activation", "act")

    def delete_attrs(self):
        delete_key(self.kwargs, "weight_ih_attr")
        delete_key(self.kwargs, "bias__ih_attr")
        delete_key(self.kwargs, "weight_hh_attr")
        delete_key(self.kwargs, "bias_hh_attr")
        delete_key(self.kwargs, "direction")
        delete_key(self.kwargs, "time_major")

    def run(self):
        return self.convert_to_tlx()
