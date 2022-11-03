# coding: utf-8
import ast
import astor
from paddle2tlx.common.mapper import *


def func_replace(node):
    code_snippet = astor.to_source(node)
    for k, v in FUNC2VARIABLE.items():  # post process
        if k in code_snippet:
            code_snippet = code_snippet.replace(k, v)
    return code_snippet
