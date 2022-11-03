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
        delete_key(self.kwargs, "initializer")
        param_list = new_param_val.replace(' ', '').split(',')
        for param in param_list:
            self.args.append(param)

    def run(self):
        if self.paddle_api_name == "paddle.fluid.param_attr.ParamAttr" and \
                self.rename_func_name("tensorlayerx.nn.initializers.random_uniform"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_paddle()


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

    def run(self):
        if self.paddle_api_name == "paddle.ParamAttr" and \
                self.rename_func_name("tensorlayerx.nn.initializers.xavier_uniform"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_paddle()
