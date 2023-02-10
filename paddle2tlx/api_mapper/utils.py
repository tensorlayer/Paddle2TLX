# coding: utf-8
import re
import inspect


def api_args2kwargs(paddle_api_name, args, first_same_attr_count):
    """ 将每个OP的args转为kwargs。

    Args:
        paddle_api_name (str): OP的类型名字。
        args (list): 参数列表。
        first_same_attr_count (int): Paddle与TensorLayerX前first_same_attr_count个完全相同的参数。
    """

    def get_default_args(obj):
        if inspect.isbuiltin(obj):
            demp_str = obj.__doc__.split("->")[0].strip()[:-1]
            demp_str = demp_str.split("(")[-1]
            demp_str_seg = demp_str.split(",")
            default_args = list()
            for seg in demp_str_seg:
                seg = seg.strip().replace("*", "")
                if seg == "":
                    continue
                if "=" in seg:
                    seg = seg.split("=")[0]
                default_args.append(seg)
            return default_args
        else:
            signature = inspect.signature(obj)
            return [k for k, v in signature.parameters.items()]

    import paddle
    obj = paddle
    for i, part in enumerate(paddle_api_name.split(".")):
        if i == 0:
            continue
        obj = getattr(obj, part)
    default_args = get_default_args(obj)
    new_kwargs = dict()
    for i, default_k in enumerate(default_args):
        if i >= first_same_attr_count and i < len(args):
            # new_kwargs[default_k] = args[i]
            if isinstance(args[i], str):
                arg_val = args[i].strip()
                if "(" in arg_val and ")" in arg_val and "int" not in arg_val:  # bug fixed
                    new_kwargs[default_k] = arg_val.replace("(", "").replace(")", "")
                else:
                    new_kwargs[default_k] = arg_val  # TODO
            else:
                new_kwargs[default_k] = args[i]
    return new_kwargs


def rename_key(kwargs, old_key, new_key):
    if old_key in kwargs:
        v = kwargs.pop(old_key)
        kwargs[new_key] = v


def delete_key(kwargs, old_key):
    if old_key in kwargs:
        kwargs.pop(old_key)


def generate_api_code(func_name, args, kwargs):
    for i, arg in enumerate(args):
        if not isinstance(args[i], str):
            args[i] = str(args[i])
    args_str = ", ".join(args)
    kwargs_str_list = list()
    for k, v in kwargs.items():
        # if isinstance(v, str):  # todo-bug: when v is variable, the generate code is converted string
        if v == "channels_first" or v == "tlx" or "tlx_pretrained_model" in str(v):
            kwargs_str_list.append("{}='{}'".format(k, v))
        else:
            kwargs_str_list.append("{}={}".format(k, v))
    kwargs_str = ", ".join(kwargs_str_list)
    if len(args_str) > 0:
        code = "{}({}, {})".format(func_name, args_str, kwargs_str)
    else:
        if func_name != "act_layer":
            code = "{}({})".format(func_name, kwargs_str)
        else:
            code = "{}".format(func_name)
    if func_name == "nn.Sequential" and len(args) > 0:
        # if args[0][0] == "[" or "(" not in args:  # todo
        # if args[0][0] != '*':  # todo
        if args[0][0] == "[":
            return code
        elif len(args) == 1:
            is_variable = bool(re.match('^[a-zA-Z_]+$', args[0]))
            if is_variable:
                return code
        begin_text = code[0:code.find("(") + 1]
        mid_text = code[code.find("(") + 1:code.rfind(")")]
        end_text = code[code.rfind(")"):]
        if args[0][0] == "(" and "OrderedDict" not in args_str:
            code = begin_text + "OrderedDict([" + mid_text + "])" + end_text
        if code.count("nn.Sequential") > 1 or args[0][0] != "(":  # when sequential has nested structure
            begin_text = code[0:code.find("(") + 1]
            mid_text = code[code.find("(") + 1:code.rfind(")")]
            end_text = code[code.rfind(")"):]
            code = begin_text + "[" + mid_text + "]" + end_text
    return code


class Mapper(object):
    def __init__(self,
                 func_name,
                 paddle_api_name,
                 args,
                 kwargs,
                 target_name=None):
        self.func_name = func_name
        self.paddle_api_name = paddle_api_name
        self.args = args
        self.kwargs = kwargs
        self.target_name = target_name

    def process_attrs(self):
        """ 更新参数。
        """
        pass

    def delete_attrs(self):
        """ 删除参数。
        """
        pass

    def check_attrs(self):
        """ 确认参数的值。
        """
        pass

    def rename_func_name(self, paddle2tlx_func_name=None):
        """ 判断是否为可变参数或者关键字参数,
            若为可变参数或者关键字参数，则替换参数名。
        """
        if paddle2tlx_func_name is not None and \
                (len(self.args) > 0 and isinstance(self.args[0], str) and self.args[0].startswith("*")) and \
                (len(self.args) > 1 and isinstance(self.args[-1], str) and self.args[-1].startswith("**")):  # or -> and
            self.func_name = paddle2tlx_func_name
            return True
        else:
            return False

    def convert_to_tlx(self):
        """ 1. 通过执行check、process、delete转换为paddle的参数；
            2. 生成paddle相关代码。
        """
        self.check_attrs()
        self.process_attrs()
        self.delete_attrs()
        return [], generate_api_code(self.func_name, self.args, self.kwargs), []

    def convert_args2kwargs(self, first_same_attr_count=0):
        """ 将args转换为kwargs。
        """
        if len(self.args) > first_same_attr_count:
            new_kwargs = api_args2kwargs(self.paddle_api_name, self.args,
                                         first_same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:first_same_attr_count]

    def run(self, paddle2tlx_func_name=None):
        """ 如果存在可变参数或者关键字参数，直接替换函数名为paddle2tlx的API；
            反之，调用convert_to_tlx。
        """
        if self.rename_func_name(paddle2tlx_func_name):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            return self.convert_to_tlx()
