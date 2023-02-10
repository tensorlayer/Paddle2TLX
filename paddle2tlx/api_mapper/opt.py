# coding: utf-8
from .utils import *


class AdamModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "learning_rate", "lr")

    def delete_attrs(self):
        delete_key(self.kwargs, "parameters")

    def run(self):
        if self.paddle_api_name == "paddle.optimizer.Adam" and self.rename_func_name("tensorlayerx.optimizer.Adam"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class varOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        # tlx:(x, axis=None, keepdims=False)
        rename_key(self.kwargs, "keepdim", "keepdims")

    def run(self):

        if self.paddle_api_name == "paddle.var" and self.rename_func_name("tensorlayerx.ops.reduce_variance"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()


class chunkOpMapper(Mapper):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, paddle_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "x", "value")
        rename_key(self.kwargs, "chunks", "num_or_size_splits")

    def run(self):

        if self.paddle_api_name == "paddle.chunk" and self.rename_func_name("tensorlayerx.split"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # self.convert_args2kwargs(1)
            return self.convert_to_tlx()
