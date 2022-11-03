# coding: utf-8
import astor
import os
import shutil
from .dependency_analyzer import analyze
from .ast_update import update
from .utils import *


def write_file(path, tree):
    code = astor.to_source(tree)
    code = code.replace("(...)", "...")
    code = add_line_continuation_symbol(code)
    f = open(path, "w")
    f.write(code)
    f.close()


def generate_dependencies(folder_path, file_dependencies):
    for name in os.listdir(folder_path):
        current_path = osp.join(folder_path, name)
        if osp.isfile(current_path) and current_path.endswith(".py"):
            if current_path in file_dependencies:
                continue
            analyze(current_path, file_dependencies)
        elif osp.isdir(current_path):
            generate_dependencies(current_path, file_dependencies)


def convert_code(folder_path, new_folder_path, file_dependencies):
    for name in os.listdir(folder_path):
        current_path = osp.join(folder_path, name)
        new_current_path = osp.join(new_folder_path, name)
        if osp.isfile(current_path) and current_path.endswith(".py"):
            print(current_path)
            root = update(current_path, file_dependencies)
            if root is not None:
                write_file(new_current_path, root)
        elif osp.isdir(current_path):
            if not osp.exists(new_current_path):
                os.makedirs(new_current_path)
            convert_code(current_path, new_current_path, file_dependencies)
        elif osp.isfile(current_path) and osp.splitext(current_path)[
                -1] in [".pdparams"]:
            continue
        elif osp.isfile(current_path) and current_path.endswith(".pyc"):
            continue
        elif osp.isdir(current_path) and current_path == "__pycache__":
            continue
        elif osp.isdir(current_path) and current_path == ".ipynb_checkpoints":
            continue
        else:
            shutil.copyfile(current_path, new_current_path)


def load_inference_model(tlx_project_path, model_name="vgg16"):
    from examples.models_tlx import TLXClassificationModel

    ModelTLX = TLXClassificationModel(tlx_project_path, model_name)
    tlx_model = ModelTLX.tlx_model

    return tlx_model


def validation(tlx_model, model_name="vgg16"):
    from examples.predict_vision import predict_tlx

    image_file = "../examples/images/dog.jpeg"
    predict_tlx(tlx_model, image_file, model_name)


# def load_inference_model(pd_project_path, tlx_project_path, model_name="vgg16"):
#     from examples.models_tlx import TLXClassificationModel
#     from examples.models_pd import PaddleClassificationModel
#
#     ModelTLX = TLXClassificationModel(tlx_project_path, model_name)
#     tlx_model = ModelTLX.tlx_model
#     ModelPaddle = PaddleClassificationModel(pd_project_path, model_name)
#     pd_model = ModelPaddle.pd_model
#
#     return tlx_model, pd_model
#
#
# def validation(tlx_model, pd_model, model_name="vgg16"):
#     """ """
#     from examples.predict_vision import calc_diff
#     from examples.predict_vision import predict_tlx, predict_pd
#
#     image_file = "../examples/images/dog.jpeg"
#     calc_diff(tlx_model, pd_model, image_file, model_name)
#     # predict_tlx(tlx_model, image_file, model_name)
#     predict_pd(pd_model, image_file, model_name)


def main(args):
    project_src_path = args.input_dir_pd
    project_dst_path = args.output_dir_tlx
    params_path = args.pretrain_model
    # model_name = args.model_name
    model_name = "mobilenetv3"  # for test
    is_save = args.save_tag

    # step1: convert paddle project to tensorlayerx & convert pretrained model file to tensorlayerx
    project_path = osp.abspath(project_src_path)
    file_dependencies = dict()
    sys.path.append(project_path)
    generate_dependencies(project_path, file_dependencies)
    if not osp.exists(project_dst_path):
        os.makedirs(project_dst_path)
    convert_code(project_path, project_dst_path, file_dependencies)

    # step2: load converted inference model
    tlx_model = load_inference_model(project_dst_path, model_name)
    # tlx_model, pd_model = load_inference_model(project_src_path, project_dst_path, model_name)

    # step3: validate the inference model of converted project
    validation(tlx_model, model_name)
    # validation(tlx_model, pd_model, model_name)
    # print('Done!')
