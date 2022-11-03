# coding: utf-8
import ast
import astor
import copy
from paddle2tlx.common.mapper import *
from .utils import get_dep_file_path


class DepInfo:
    """
    依赖包信息。
    PD_FROM代表paddle from信息的字符串，例如：paddle；
    TLX_FROM代表tensorlayerx from信息的字符串，例如：tensorlayerx；
    PD_IMPORT代表paddle import信息系的字符串，例如：nn.functional；
    TLX_IMPORT代表tensorlayerx import信息系的字符串，例如：nn.functional；
    AS代表as信息的字符串，例如：F；
    PD_DEPENDENCY代表由PD_FROM、PD_IMPORT、AS三者组成的字符串，例如：from paddle import nn.functional as F。
    TLX_DEPENDENCY代表由TLX_FROM、TLX_IMPORT、AS三者组成的字符串，例如：。
    """
    PD_FROM = None
    TLX_FROM = None
    PD_IMPORT = None
    TLX_IMPORT = None
    AS = None
    PD_DEPENDENCY = None
    TLX_DEPENDENCY = None


class AstUpdater(ast.NodeVisitor):
    """ 更新ast树，将ast树中Paddle相关的节点转为TensorLayerX相关的节点。
    Args:
        py_file_path (str): python文件的绝对值路径。
        file_dependencies (dict): 当前已经统计的依赖信息，key为python文件的绝对值路径，
                                value为key文件所对应的依赖信息组成的list。
    """

    def __init__(self, py_file_path, file_dependencies):
        self.py_file_path = py_file_path
        self.root = ast.parse(open(py_file_path, "rb").read())
        self.file_dependencies = file_dependencies
        self.scopes_and_dependencies = list()  # 作用域和依赖组成的stack
        self.nodes = list()  # ast节点组成的stack
        self.no_support_apis = list()  # 不支持的API列表
        # self.is_import_paddle2tlx = False  # 是否添加import paddle2tlx
        self.is_import_tlx = True  # 是否添加import tensorlayerx as tlx
        self.is_import_pd = True
        # self.is_import_backend = True
        self.is_import_restore = True
        self.visit_import = list()  # 已导包模块, 用于去重

    def _get_scope_node(self):
        """ 获取当前节点的作用域。
        """
        scope_node = None
        for i in range(len(self.scopes_and_dependencies)):
            i = -(i + 1)
            sd = self.scopes_and_dependencies[i]
            if not isinstance(sd, DepInfo) and not isinstance(sd, ast.Assign):
                scope_node = sd
                break
        return scope_node

    def _get_current_index(self, scope_node, node):
        """ 获取当前节点在其作用域中的索引序号。
        """
        current_id = 0
        for i, n in enumerate(scope_node.body):
            if node == n:
                current_id = i
                break
        return current_id

    def _get_father_node(self):
        """ 获取父节点。
        """
        return self.nodes[-2]

    def _get_import_module(self):
        """ 获取已导包模块
        """
        import_info = []
        for i in range(len(self.scopes_and_dependencies)):
            dep_info = self.scopes_and_dependencies[i]
            if isinstance(dep_info, DepInfo):
                pass
        return import_info

    def _get_complete_api(self, api_part_name):  # TODO - 需修改
        """ 根据部分api名字获取Paddle的api全名。
            情况1：依赖是DepInfo，但其TLX_IMPORT为None（非Paddle的依赖），则tlx_api为None。
            情况2：依赖是DepInfo，且DepInfo的部分Paddle属性以“paddle”开头，则tlx_api为完整api。
            情况3：依赖是ast.Assign节点，则tlx_api为None。
        """
        paddle_api = None
        dep_info = None
        if api_part_name in CUSTOM_API.keys():  # TODO
            return api_part_name, dep_info
        if api_part_name is None:
            return paddle_api, dep_info
        for i in range(len(self.scopes_and_dependencies)):
            i = -(i + 1)
            dep_info = self.scopes_and_dependencies[i]
            if isinstance(dep_info, DepInfo):
                if dep_info.PD_IMPORT is None:
                    continue
                if (dep_info.PD_FROM is not None and "paddle" in dep_info.PD_FROM) or \
                        (dep_info.PD_IMPORT is not None and "paddle" in dep_info.PD_IMPORT):
                    replace_str = None
                    if dep_info.AS is not None and api_part_name.startswith(
                            dep_info.AS + "."):
                        replace_str = dep_info.AS
                    elif dep_info.AS is None and api_part_name.startswith(
                            dep_info.PD_IMPORT):
                        replace_str = dep_info.PD_IMPORT
                    if replace_str is not None:
                        paddle_api = api_part_name.replace(
                            replace_str, dep_info.PD_DEPENDENCY, 1)
                        if "paddle2tlx" in paddle_api:  # TODO
                            # 说明当前节点是插入的已经替换过的node
                            paddle_api = None
                        break
            elif isinstance(dep_info, ast.Assign):
                is_customized = False
                for s in astor.to_source(dep_info.targets[0]).split(","):
                    if api_part_name.split(".")[0] == s.strip():
                        is_customized = True
                        break
                if is_customized:
                    break
        return paddle_api, dep_info

    def _rename(self, name, dep_info, paddle_api, tlx_api):
        """ 对函数名进行重命名。
            例如：将nn.Conv2D替换为nn.Conv2d。
        """
        if dep_info is None:
            return tlx_api
        if dep_info.TLX_IMPORT is None:  # TODO
            return tlx_api
        paddle_api_seg = paddle_api.split(dep_info.PD_IMPORT)
        if tlx_api.startswith(dep_info.TLX_IMPORT + ".") or \
                tlx_api.endswith("." + dep_info.TLX_IMPORT) or  \
                "." + dep_info.TLX_IMPORT + "." in tlx_api:
            tlx_api_seg = tlx_api.split(dep_info.TLX_IMPORT)
            if dep_info.AS is None:
                name = name.replace(dep_info.PD_IMPORT + paddle_api_seg[-1],
                                    dep_info.TLX_IMPORT + tlx_api_seg[-1])
            else:

                name = name.replace(paddle_api_seg[-1], tlx_api_seg[-1])
        # elif "paddle2tlx." in tlx_api:
        #     name = "paddle2tlx." + tlx_api.split("paddle2tlx.")[-1]
        #     self.is_import_paddle2tlx = True
        else:  # TODO
            name = tlx_api
        return name

    def run(self):
        self.scopes_and_dependencies.append(self.root)
        self.visit(self.root)
        for i, node in enumerate(self.root.body):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                # if astor.to_source(node).strip() not in self.visit_import:
                #     self.visit_import.append(astor.to_source(node).strip())
                # else:
                #     continue
                if isinstance(node, ast.ImportFrom) and node.module in HEADER_IMPORT:
                    continue
                if self.is_import_restore:
                    self.root.body.insert(i,
                                          ast.parse("from examples.utils.load_model_tlx import restore_model").body[0])
                # if self.is_import_restore:
                #     self.root.body.insert(i,
                #                           ast.parse("from examples.utils.load_model_tlx import load_model").body[0])
                if self.is_import_pd:
                    self.root.body.insert(i, ast.parse("import paddle").body[0])
                if self.is_import_tlx:
                    self.root.body.insert(i, ast.parse("import tensorlayerx as tlx").body[0])
                break

    def visit(self, node):
        self.nodes.append(node)
        out = super(AstUpdater, self).visit(node)
        self.nodes.pop()
        if out is not None:
            return out
        else:
            # 出现字符串或者if等节点需要返回字符串
            try:
                return astor.to_source(node)
            except Exception:
                return None

    def visit_ImportFrom(self, node):
        """ 1. 遍历子节点。
            2. 将当前from依赖中的多个import拆分成多个import。
                例如：from paddle import nn, utils 这个node
                     拆分为：node1：from paddle import nn
                            node2：from paddle import utils
                拆分原因：
                    在 tensorlayerx 中父依赖包可能不一致。
        """
        scope_node = self._get_scope_node()
        current_id = self._get_current_index(scope_node, node)
        scope_node.body.pop(current_id)
        son_nodes = node.names
        for i, son_node in enumerate(son_nodes):
            copy_node = copy.deepcopy(node)
            copy_node.names = [son_node]
            if i == 0:
                is_remove = self.visit_alias(son_node, copy_node, node.module,
                                             node.level)  # TODO
                if not is_remove:
                    scope_node.body.insert(current_id, copy_node)
            else:
                scope_node.body.insert(current_id + i, copy_node)

    def visit_Import(self, node):
        """ 遍历子节点。
        """
        son_nodes = getattr(node, "names")
        for son_node in son_nodes:
            self.visit_alias(son_node, node)

    def visit_alias(self,
                    node,
                    father_node=None,
                    from_name=None,
                    from_level=None):
        """ 构建DepInfo并将其放入scopes_and_dependencies。
            如果import字符串为“*”，获取依赖包所在文件的依赖信息并转换为DepInfo加入当前的scopes_and_dependencies；
            反之，直接在scopes_and_dependencies中加入DepInfo。
        """
        is_remove = False
        dep_info = DepInfo()
        dep_info.PD_FROM = from_name
        dep_info.PD_IMPORT = getattr(node, "name")
        dep_info.AS = getattr(node, "asname", None)
        if dep_info.PD_IMPORT == "*":
            import_file_path = get_dep_file_path(self.py_file_path, from_level, from_name)
            paddle_dependencies = self.file_dependencies[import_file_path]
            for paddle_dep_info in paddle_dependencies:
                current_dep_info = DepInfo()
                if not isinstance(paddle_dep_info, str):
                    current_dep_info.PD_FROM = paddle_dep_info.FROM
                    current_dep_info.PD_IMPORT = paddle_dep_info.IMPORT
                    current_dep_info.AS = paddle_dep_info.AS
                    current_dep_info.PD_DEPENDENCY = paddle_dep_info.DEPENDENCY
                    if "paddle" in current_dep_info.PD_DEPENDENCY:
                        if current_dep_info.PD_DEPENDENCY in API_MAPPER:
                            current_dep_info.TLX_DEPENDENCY = \
                                    API_MAPPER[current_dep_info.PD_DEPENDENCY][0]
                            if current_dep_info.PD_DEPENDENCY == "paddle" and \
                                    isinstance(self._get_scope_node(), ast.Module):
                                self.is_import_tlx = False
                            if current_dep_info.PD_FROM is not None:
                                seg = current_dep_info.TLX_DEPENDENCY.split(".")
                                current_dep_info.TLX_IMPORT = seg[-1]
                                current_dep_info.TLX_FROM = \
                                        current_dep_info.TLX_DEPENDENCY.replace("." + seg[-1], "")
                            else:
                                current_dep_info.TLX_IMPORT = \
                                current_dep_info.TLX_DEPENDENCY
                        elif current_dep_info.PD_DEPENDENCY in REMOVE_API:
                            scope_node = self._get_scope_node()
                            for i, n in enumerate(scope_node.body):
                                if father_node == n:
                                    scope_node.body[i] = ast.parse("\n").body
                            is_remove = True
                        else:
                            self.no_support_apis.append(
                                current_dep_info.PD_DEPENDENCY)
                else:
                    current_dep_info.TLX_DEPENDENCY = paddle_dep_info
                self.scopes_and_dependencies.append(current_dep_info)
            return is_remove
        dependency_str_list = list()
        if dep_info.PD_FROM is None and from_level is not None:
            dependency_str_list.append("." * from_level)
        elif dep_info.PD_FROM is not None:
            dependency_str_list.append(dep_info.PD_FROM)
        dependency_str_list.append(dep_info.PD_IMPORT)
        dep_info.PD_DEPENDENCY = ".".join(dependency_str_list)
        if dep_info.PD_DEPENDENCY.startswith("paddle"):
            if dep_info.PD_DEPENDENCY in API_MAPPER:
                dep_info.TLX_DEPENDENCY = API_MAPPER[dep_info.PD_DEPENDENCY][0]
                if dep_info.PD_DEPENDENCY == "paddle":
                    # self.is_import_tlx = False  # TODO
                    pass
                if dep_info.PD_FROM is not None:
                    seg = dep_info.TLX_DEPENDENCY.split(".")
                    setattr(node, "name", seg[-1])
                    setattr(father_node, "module",
                            dep_info.TLX_DEPENDENCY.replace("." + seg[-1], ""))
                    dep_info.TLX_IMPORT = seg[-1]
                    dep_info.TLX_FROM = dep_info.TLX_DEPENDENCY.replace(
                        "." + seg[-1], "")
                else:
                    setattr(node, "name", dep_info.TLX_DEPENDENCY)
                    dep_info.TLX_IMPORT = dep_info.TLX_DEPENDENCY
            elif dep_info.PD_DEPENDENCY in REMOVE_API:
                scope_node = self._get_scope_node()
                for i, n in enumerate(scope_node.body):
                    if father_node == n:
                        scope_node.body[i] = ast.parse("\n").body
                is_remove = True
            elif dep_info.PD_DEPENDENCY.startswith("paddle"):
                self.no_support_apis.append(dep_info.PD_DEPENDENCY)
        else:
            dep_info.TLX_DEPENDENCY = dep_info.PD_DEPENDENCY
            if dep_info.PD_DEPENDENCY in REMOVE_API:  # TODO
                scope_node = self._get_scope_node()
                for i, n in enumerate(scope_node.body):
                    if father_node == n:
                        scope_node.body[i] = ast.parse("\n").body
                is_remove = True
        self.scopes_and_dependencies.append(dep_info)
        return is_remove

    def visit_Name(self, node):
        """ 获取字符串名字。
        """
        paddle_api, dep_info = self._get_complete_api(getattr(node, "id"))
        father_node = self._get_father_node()
        if paddle_api in API_MAPPER:
            tlx_api = API_MAPPER[paddle_api][0]
            if isinstance(father_node, ast.Call) and getattr(
                    father_node.func, "id", None) in ("getattr", "setattr", "hasattr"):
                tlx_api = self._rename(tlx_api, dep_info, paddle_api, tlx_api)
                for i, arg_node in enumerate(father_node.args):
                    if astor.to_source(arg_node).strip() == getattr(node, "id"):
                        father_node.args[i] = ast.parse(tlx_api).body[0].value
        return getattr(node, "id")

    def visit_Attribute(self, node):
        """ 对属性字符串满足以下5种情况时进行替换：
            情况1 —— Class A(nn.Layer)：将nn.Module替换为nn.Module；
            情况2 —— a = (1, 2, nn.Layer)：将nn.Module替换为nn.Module；
            情况3 —— def a() -> paddle.Tensor：将paddle.Tensor替换为tlx.ops.convert_to_tensor；
            情况4 —— def a(x: paddle.Tensor)：将paddle.Tensor替换为tlx.ops.convert_to_tensor；
            情况5 —— isinstance(a, nn.Layer)：将nn.Layer替换为nn.Module;
        """
        if node.lineno == 85:
            pass
            print('...')
        value_node = node.value
        attr = node.attr  # Layer
        name = self.visit(value_node)  # nn
        attr_str = name + "." + attr  # nn.Layer
        paddle_api, dep_info = self._get_complete_api(attr_str)
        father_node = self._get_father_node()
        if paddle_api in API_MAPPER:
            tlx_api = API_MAPPER[paddle_api][0]
            if isinstance(father_node, ast.ClassDef):
                attr_str = self._rename(attr_str, dep_info, paddle_api, tlx_api)
                if node in father_node.bases:
                    father_node.bases[0] = ast.parse(attr_str).body[0].value
                return attr_str  # TODO - 'paddle.nn.Layer'
            elif isinstance(father_node, ast.arguments):
                attr_str = self._rename(attr_str, dep_info, paddle_api, tlx_api)
                for i, default_n in enumerate(father_node.defaults):
                    if default_n == node:
                        father_node.defaults[i] = ast.parse(attr_str).body[0].value
                return attr_str
            elif isinstance(father_node, ast.Tuple):
                tlx_api = self._rename(tlx_api, dep_info, paddle_api, tlx_api)
                for i, elts_node in enumerate(father_node.elts):
                    if astor.to_source(elts_node).strip() == attr_str:
                        father_node.elts[i] = ast.parse(tlx_api).body[0].value
                return tlx_api
            elif isinstance(father_node, ast.FunctionDef):
                tlx_api = self._rename(tlx_api, dep_info, paddle_api, tlx_api)
                father_node.returns = ast.parse(tlx_api).body[0].value
                return tlx_api
            elif isinstance(father_node, ast.arg):
                attr_str = self._rename(attr_str, dep_info, paddle_api, tlx_api)
                father_node.annotation = ast.parse(attr_str).body[0].value
                return attr_str
            elif isinstance(father_node, ast.Call) and getattr(
                    father_node.func, "id", None) == "isinstance":
                tlx_api = self._rename(tlx_api, dep_info, paddle_api, tlx_api)
                for i, arg_node in enumerate(father_node.args):
                    if astor.to_source(arg_node).strip() == attr_str:
                        father_node.args[i] = ast.parse(tlx_api).body[0].value
                return tlx_api
            elif isinstance(father_node, ast.Assign):  # TODO
                # var_name = astor.to_source(father_node.targets[0]).strip()
                # var_value = astor.to_source(node).strip()
                # for k, v in CUSTOM_ASSIGN_MAP.items():
                #     if k == var_name and v == var_value:
                #         return
                attr_str = self._rename(attr_str, dep_info, paddle_api, tlx_api)
                father_node.value = ast.parse(attr_str).body[0].value
                return attr_str
        elif paddle_api in REMOVE_API:
            if isinstance(father_node, (ast.Assign, ast.If, ast.FunctionDef, ast.ClassDef, ast.Call)):
                scope_node = self._get_scope_node()
                for i, n in enumerate(scope_node.body):
                    if father_node == n:
                        scope_node.body.pop(i)
                        return None
            elif isinstance(father_node, ast.BoolOp):
                for i, n in enumerate(father_node.values):
                    if node == n:
                        father_node.values[i] = ast.parse("False").body[0].value
                        return None
        else:
            if isinstance(paddle_api, str) and paddle_api.startswith(
                    "paddle"
            ) and "(" not in paddle_api and "[" not in paddle_api:
                if not isinstance(father_node, ast.Attribute):
                    self.no_support_apis.append(paddle_api)
        return attr_str

    def visit_arguments(self, node):
        """ 修改函数关键字参数默认api
        """
        for param_val in node.defaults:
            if hasattr(param_val, "id"):  # case 1
                for k, v in REPLACE_API.items():
                    if param_val.id == k:
                        param_val.id = v
            if hasattr(param_val, "attr"):
                if not hasattr(param_val, "value"):  # case 2
                    for k, v in REPLACE_API.items():  # not optimal solution
                        if param_val.attr == k:
                            param_val.attr = v
                else:  # case 3
                    old_api = astor.to_source(param_val).strip()
                    if old_api in REPLACE_API.keys():
                        new_api_attr = REPLACE_API[old_api].split(".")[-1]
                        right_idx = REPLACE_API[old_api].rfind(".")
                        new_api_value = REPLACE_API[old_api][:right_idx]
                        param_val.attr = new_api_attr
                        param_val.value = ast.parse(new_api_value).body[0].value

    def visit_Num(self, node):
        """ 返回数值。
        """
        return getattr(node, "n")

    def visit_keyword(self, node):
        """ 修改函数调用中关键字参数默认值，返回键值对。
            【注意】当value是API_MAPPER中的key时，需要替换为API_MAPPER中对应的Paddle API。
        """
        key = getattr(node, "arg")
        value_node = getattr(node, "value")
        if value_node.lineno == 143:
            pass
            print('1')

        if isinstance(value_node, ast.Call):
            param_v = astor.to_source(value_node)
            if param_v.strip() in REPLACE_API.keys():
                for k, v in REPLACE_API.items():
                    if k in param_v:
                        old_api_name = param_v[param_v.find("=") + 1:]
                        new_api_name = v
                        if "\n" in old_api_name:
                            new_api_name = v + "\n"
                        return {key: new_api_name}
        if isinstance(value_node, ast.Attribute):
            param_v = astor.to_source(value_node)
            if param_v.strip() in REPLACE_API.keys():
                for k, v in REPLACE_API.items():
                    if k in param_v:
                        new_api_name = v
                        node.value = ast.parse(new_api_name).body[0].value

        value = self.visit(value_node)
        if value in API_MAPPER:
            value = API_MAPPER[value][0]
        elif isinstance(value, str) and value.startswith(
                "paddle") and "(" not in value and "[" not in value:
            self.no_support_apis.append(value)
        return {key: value}

    def visit_Tuple(self, node):
        """ 返回tuple。
        """
        elts_nodes = getattr(node, "elts")
        elts = list()
        for elts_node in elts_nodes:
            elt = self.visit(elts_node)
            elts.append(elt if isinstance(elt, str) else str(elt))
        elts = "({})".format(", ".join(elts))
        return elts

    def visit_Assign(self, node):
        """ 1. 将Assign节点加入scopes_and_dependencies；
            2. 遍历Assign节点的子节点。
        """
        self.scopes_and_dependencies.append(node)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        """ 返回累乘数值对象结果
        """
        expr_str = astor.to_source(node).strip()
        try:
            if "(" in expr_str and ")" in expr_str:
                expr_str = expr_str[1:-1]
                num_list = expr_str.replace(" ", "").split('*')
                res = 1
                for num in num_list:
                    num = int(num)
                    res *= num
                return res
        except Exception as e:
            pass

    def visit_Call(self, node):
        """ 1. 获取原始函数名并更新为新的函数名。
            2. 获取args和kwargs。
            3. 根据API_MAPPER映射需要更新的操作，对参数进行处理。
            4. 如果有前缀代码和后缀代码，则需要添加相应节点。
        """
        # TODO - for debug
        if node.lineno == 85:  # 42
            pass
            print('...')

        # 获取函数名
        func_node = node.func
        if isinstance(func_node, ast.Attribute) and isinstance(func_node.value,
                                                               ast.Call):
            func_name = None
        else:
            func_name = self.visit(func_node)
        paddle_api, dep_info = self._get_complete_api(func_name)
        if paddle_api is None:
            self.generic_visit(node)
            return
        if paddle_api not in API_MAPPER:
            if paddle_api.startswith(
                    "paddle"
            ) and "[" not in paddle_api and "(" not in paddle_api:
                self.no_support_apis.append(paddle_api)
            return
        tlx_api = API_MAPPER[paddle_api][0]
        if func_name in CUSTOM_API.keys():  # TODO
            func_name = tlx_api
        else:
            func_name = self._rename(func_name, dep_info, paddle_api, tlx_api)
        setattr(node, "func", ast.parse(func_name).body[0].value)

        # 获取args
        args_nodes = getattr(node, "args")
        args_list = list()
        for args_node in args_nodes:
            args_list.append(self.visit(args_node))

        # 获取keywords
        keywords_nodes = getattr(node, "keywords")
        kw_dict = dict()
        for keywords_node in keywords_nodes:
            if list(self.visit(keywords_node).keys())[0] is None:
                args_list.append("**{}".format(
                    list(self.visit(keywords_node).values())[0]))
            else:
                kw_dict.update(self.visit(keywords_node))

        if API_MAPPER[paddle_api][1] is None:
            return

        target_name = None
        father_node = self._get_father_node()
        if isinstance(father_node, ast.Assign):
            target_node = father_node.targets[0]
            target_name = self.visit(target_node)
        mapper = API_MAPPER[paddle_api][1](func_name, paddle_api, args_list,
                                            kw_dict, target_name)
        prefix_insert_codes, new_code, suffix_insert_codes = mapper.run()
        scope_node = self._get_scope_node()
        if isinstance(ast.parse(new_code).body[0], ast.Assign):
            node_index = self._get_current_index(scope_node, node)
            scope_node.body[node_index] = ast.parse(
                new_code.replace("\n", "")).body[0]
        else:
            new_call_node = ast.parse(new_code).body[0].value
            setattr(node, "func", new_call_node.func)  # 修改了fun_name
            setattr(node, "args", new_call_node.args)
            setattr(node, "keywords", new_call_node.keywords)
        for i, n in enumerate(scope_node.body):
            if father_node == n:
                for code in prefix_insert_codes:
                    scope_node.body.insert(
                        i, ast.parse(code.replace("\n", "")).body[0])
                    i += 1
                break
        for i, n in enumerate(scope_node.body):
            if father_node == n:
                j = i + 1
                for code in suffix_insert_codes:
                    scope_node.body.insert(
                        j, ast.parse(code.replace("\n", "")).body[0])
                    j += 1
                break

    def visit_Subscript(self, node):
        value_node = node.value
        value_name = self.visit(value_node)
        paddle_api, dep_info = self._get_complete_api(value_name)
        if paddle_api in API_MAPPER:
            tlx_api = API_MAPPER[paddle_api][0]
            value_name = self._rename(value_name, dep_info, paddle_api,
                                      tlx_api)
            node.value = ast.parse(value_name).body[0]
        else:  # 不支持api
            if isinstance(paddle_api, str) and paddle_api.startswith(
                    "paddle") and "(" not in paddle_api:
                self.no_support_apis.append(paddle_api)
        self.visit(node.slice)
        self.visit(node.ctx)

    def visit_FunctionDef(self, node):
        """ 1. 将FunctionDef节点加入scopes_and_dependencies；
            2. 遍历FunctionDef节点的子节点；
            3. 去除scopes_and_dependencies中FunctionDef节点以及之后的节点。
        """
        self.scopes_and_dependencies.append(node)
        self.generic_visit(node)
        last_node = self.scopes_and_dependencies.pop(-1)
        while not isinstance(last_node, ast.FunctionDef):
            last_node = self.scopes_and_dependencies.pop(-1)

    def visit_ClassDef(self, node):
        """ 1. 将ClassDef节点加入scopes_and_dependencies；
            2. 遍历ClassDef节点的子节点；
            3. 去除scopes_and_dependencies中ClassDef节点以及之后的节点。
        """
        self.scopes_and_dependencies.append(node)
        self.generic_visit(node)
        last_node = self.scopes_and_dependencies.pop(-1)
        while not isinstance(last_node, ast.ClassDef):
            last_node = self.scopes_and_dependencies.pop(-1)

    def visit_If(self, node):
        """ 1. 将If节点加入scopes_and_dependencies；
            2. 遍历If节点的子节点；
            3. 去除scopes_and_dependencies中If节点以及之后的节点。
        """
        self.scopes_and_dependencies.append(node)
        self.generic_visit(node)
        last_node = self.scopes_and_dependencies.pop(-1)
        while not isinstance(last_node, ast.If):
            last_node = self.scopes_and_dependencies.pop(-1)

    def visit_While(self, node):
        """ 1. 将While节点加入scopes_and_dependencies；
            2. 遍历Try节点的子节点；
            3. 去除scopes_and_dependencies中Try节点以及之后的节点。
        """
        self.scopes_and_dependencies.append(node)
        self.generic_visit(node)
        last_node = self.scopes_and_dependencies.pop(-1)
        while not isinstance(last_node, ast.While):
            last_node = self.scopes_and_dependencies.pop(-1)

    def visit_Try(self, node):
        """ 1. 将Try节点加入scopes_and_dependencies；
            2. 遍历Try节点的子节点；
            3. 去除scopes_and_dependencies中Try节点以及之后的节点。
        """
        self.scopes_and_dependencies.append(node)
        self.generic_visit(node)
        last_node = self.scopes_and_dependencies.pop(-1)
        while not isinstance(last_node, ast.Try):
            last_node = self.scopes_and_dependencies.pop(-1)

    def visit_ExtSlice(self, node):
        """ 将Index节点替换为Num。
        """
        dim_nodes = node.dims
        for i, dim_node in enumerate(dim_nodes):
            if isinstance(dim_node, ast.Index):
                dim_nodes[i] = dim_node.value
            else:
                self.visit(dim_node)

    def visit_Str(self, node):
        """ 修改模型参数的后缀名。
        """
        # setattr(node, "s", node.s.replace(".pdparams", ".npz"))
        pass


def update(py_file_path, file_dependencies):
    updater = AstUpdater(py_file_path, file_dependencies)
    updater.run()
    if len(updater.no_support_apis) > 0:
        print("Can not convert the file {}.".format(py_file_path))
        print("The unsupported packages or operators are: [{}].".format(
            ", ".join(set(updater.no_support_apis))))
        return None
    else:
        return updater.root
