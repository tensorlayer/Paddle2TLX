# coding: utf-8
from .utils import *


class ParamAttrRandomMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        new_param_val = ''
        for k, v in self.kwargs.items():
            if k == "initializer":
                temp = v.strip()
                new_param_val = temp[temp.find("(")+1:-1]
        param_list = new_param_val.replace(' ', '').split(',')
        for param in param_list:
            self.args.append(param)

    def delete_attrs(self):
        delete_key(self.kwargs, "initializer")

    def run(self):
        if self.paddle_api_name == "paddle.fluid.param_attr.ParamAttr" and \
                self.rename_func_name("tensorlayerx.nn.initializers.random_uniform"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class ParamAttrXavierMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        pass

    def delete_attrs(self):
        delete_key(self.kwargs, "regularizer")  # clas
        delete_key(self.kwargs, "initializer")  # det
        delete_key(self.kwargs, "learning_rate")
        delete_key(self.kwargs, "trainable")

    def run(self):
        if self.paddle_api_name == "paddle.ParamAttr" and \
                self.rename_func_name("tensorlayerx.nn.initializers.xavier_uniform"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class RandomNormalMapper(Mapper):
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
        if self.paddle_api_name == "paddle.nn.initializer.Normal" and \
                self.rename_func_name("tensorlayerx.nn.initializers.random_normal"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


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
            return self.convert_to_tlx()


class TruncNormalMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        tlx2pd_weight = {
            'biases': 'bias',
            'weights': 'weight',
        }
        if len(self.args) == 1:
            param_val = self.args[0]
            for k, v in tlx2pd_weight.items():
                if v in param_val:
                    param_val = param_val.replace(v, k)
            self.args = []
            self.args.append(param_val)

    def run(self):
        if self.paddle_api_name == "paddle.nn.initializer.TruncatedNormal" and \
                self.rename_func_name("tensorlayerx.nn.initializers.TruncatedNormal"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            return self.convert_to_tlx()


class NormalCustomMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        tlx2pd_weight = {
            'filters': 'weight',  # conv2d
        }
        if len(self.args) == 1:
            param_val = self.args[0]
            for k, v in tlx2pd_weight.items():
                if v in param_val:
                    param_val = param_val.replace(v, k)
            self.args = []
            self.args.append(param_val)

    def run(self):
        return self.convert_to_tlx()


class CreateParameterMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        if len(self.args) == 1:
            self.kwargs["shape"] = self.args[0]
            self.args = []  # clear
        delete_key(self.kwargs, "attr")

    def run(self):
        if self.paddle_api_name == "self.create_parameter" and \
                self.rename_func_name("self.create_parameter"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()
