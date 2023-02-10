# coding: utf-8
import os
import shutil
from .dependency_analyzer import analyze
from .ast_update import update
from .utils import *
from .config import FILE_NOT_CONVERT


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
            if current_path.split("\\")[-1] in FILE_NOT_CONVERT or current_path.split("/")[-1] in FILE_NOT_CONVERT\
                    or current_path.split("\\")[-1] == "__init__.py" or current_path.split("/")[-1] == "__init__.py":
                shutil.copyfile(current_path, new_current_path)
                continue
            root = update(current_path, file_dependencies)
            if root is not None:
                write_file(new_current_path, root)
        elif osp.isdir(current_path):
            if not osp.exists(new_current_path):
                os.makedirs(new_current_path)
            convert_code(current_path, new_current_path, file_dependencies)
        elif osp.isfile(current_path) and osp.splitext(current_path)[
                -1] in [".pdparams"]:
            shutil.copyfile(current_path, new_current_path)
            continue
        elif osp.isfile(current_path) and current_path.endswith(".pyc"):
            continue
        elif osp.isdir(current_path) and current_path.endswith("__pycache__"):
            continue
        elif osp.isdir(current_path) and current_path.endswith(".ipynb_checkpoints"):
            continue
        else:
            shutil.copyfile(current_path, new_current_path)


def load_inference_model_tlx(tlx_project_path, model_name="vgg16", model_type="clas"):
    if model_type == "clas":
        from examples.models_clas_tlx import TLXClassificationModel
        ModelTLX = TLXClassificationModel(tlx_project_path, model_name)
        tlx_model = ModelTLX.tlx_model
        return tlx_model
    elif model_type == "seg":
        from examples.models_seg_tlx import TLXSegmentationModel
        ModelTLX = TLXSegmentationModel(tlx_project_path, model_name)  # load in first
        tlx_model = ModelTLX.tlx_model
        return tlx_model
    elif model_type == "det":
        from examples.models_det_tlx import TLXDetectionModel
        ModelTLX = TLXDetectionModel(tlx_project_path, model_name)
        tlx_model = ModelTLX.tlx_model
        return tlx_model
    elif model_type == "cdet":
        from examples.models_cdet_tlx import TLXChangeDetectionModel
        ModelTLX = TLXChangeDetectionModel(tlx_project_path, model_name)
        tlx_model = ModelTLX.tlx_model
        return tlx_model
    elif model_type == "nlp":
        from examples.models_nlp_tlx import TLXClassificationModel
        ModelTLX = TLXClassificationModel(tlx_project_path, model_name)
        tlx_model = ModelTLX.tlx_model
        return tlx_model


def validation_tlx(tlx_model, model_name="vgg16", model_type="clas"):
    if model_type == "clas":
        from examples.predict_clas import predict_tlx
        predict_tlx(tlx_model, model_name)
    elif model_type == "seg":
        from examples.predict_seg import predict_tlx
        predict_tlx(tlx_model, model_name)
    elif model_type == "det":
        from examples.predict_det import predict_tlx
        predict_tlx(tlx_model, model_name)
    elif model_type == "cdet":
        from examples.predict_cdet import predict_tlx
        predict_tlx(tlx_model, model_name)
    elif model_type == "nlp":
        from examples.predict_nlp import predict_tlx
        predict_tlx(tlx_model, model_name)


# def load_inference_model(pd_project_path, tlx_project_path, model_name="vgg16", model_type="clas"):
#     if model_type == "clas":
#         from paddle2tlx.examples.models_clas_tlx import TLXClassificationModel
#         from paddle2tlx.examples.models_clas_pd import PaddleClassificationModel
#         ModelTLX = TLXClassificationModel(tlx_project_path, model_name)  # load in first
#         tlx_model = ModelTLX.tlx_model
#         ModelPaddle = PaddleClassificationModel(pd_project_path, model_name)
#         pd_model = ModelPaddle.pd_model
#         return tlx_model, pd_model
#     elif model_type == "seg":
#         from paddle2tlx.examples.models_seg_tlx import TLXSegmentationModel
#         from paddle2tlx.examples.models_seg_pd import PaddleSegmentationModel
#         ModelTLX = TLXSegmentationModel(tlx_project_path, model_name)  # load in first
#         tlx_model = ModelTLX.tlx_model
#         ModelPaddle = PaddleSegmentationModel(pd_project_path, model_name)
#         pd_model = ModelPaddle.pd_model
#         return tlx_model, pd_model
#     elif model_type == "det":
#         from paddle2tlx.examples.models_det_tlx import TLXDetectionModel
#         from paddle2tlx.examples.models_det_pd import PaddleDetectionModel
#         ModelTLX = TLXDetectionModel(tlx_project_path, model_name)
#         tlx_model = ModelTLX.tlx_model
#         ModelPaddle = PaddleDetectionModel(tlx_project_path, model_name)
#         pd_model = ModelPaddle.pd_model
#         return tlx_model, pd_model
#     elif model_type == "cdet":
#         from paddle2tlx.examples.models_cdet_tlx import TLXChangeDetectionModel
#         from paddle2tlx.examples.models_cdet_pd import PaddleChangeDetectionModel
#         ModelTLX = TLXChangeDetectionModel(tlx_project_path, model_name)
#         tlx_model = ModelTLX.tlx_model
#         ModelPaddle = PaddleChangeDetectionModel(tlx_project_path, model_name)
#         pd_model = ModelPaddle.pd_model
#         return tlx_model, pd_model
#
#
# def validation(tlx_model, pd_model, model_name="vgg16", model_type="clas"):
#     """ """
#     if model_type == "clas":
#         from paddle2tlx.examples.predict_clas import calc_diff
#         from paddle2tlx.examples.predict_clas import predict_tlx, predict_pd
#         calc_diff(tlx_model, pd_model, model_name)
#         predict_tlx(tlx_model, model_name)
#         predict_pd(pd_model, model_name)
#     elif model_type == "seg":
#         from paddle2tlx.examples.predict_seg import predict_tlx, predict_pd, calc_diff
#         result_tlx = predict_tlx(tlx_model, model_name)
#         result_pd = predict_pd(pd_model, model_name)
#         calc_diff(result_tlx, result_pd, model_name)
#     elif model_type == "det":
#         from paddle2tlx.examples.predict_det import predict_tlx, predict_pd, calc_diff
#         result_tlx = predict_tlx(tlx_model, model_name)
#         result_pd = predict_pd(pd_model, model_name)
#         calc_diff(result_tlx, result_pd, model_name)
#     elif model_type == "cdet":
#         from paddle2tlx.examples.predict_cdet import predict_tlx, predict_pd, calc_diff
#         result_tlx = predict_tlx(tlx_model, model_name)
#         result_pd = predict_pd(pd_model, model_name)
#         calc_diff(result_tlx, result_pd, model_name)


def main(args):
    project_src_path = args.input_dir_pd
    project_dst_path = args.output_dir_tlx
    model_name = args.model_name
    model_type = args.model_type
    is_save = args.save_tag
    params_path = args.pretrain_model
    # project_src_path = "D:/Program/Work/ProjectSToneHQ/model_convert_tools/convert_test/paddlers"
    # project_dst_path = "D:/Program/Work/ProjectSToneHQ/model_convert_tools/convert_test/paddleclas"
    # if model_name:
    #     project_src_path = os.path.join(project_src_path, model_name)
    #     project_dst_path = os.path.join(project_dst_path, model_name)

    # step1: convert paddle project to tensorlayerx & convert pretrained model file to tensorlayerx
    project_path = osp.abspath(project_src_path)
    file_dependencies = dict()
    sys.path.append(project_path)
    generate_dependencies(project_path, file_dependencies)
    if not osp.exists(project_dst_path):
        os.makedirs(project_dst_path)
    convert_code(project_path, project_dst_path, file_dependencies)

    if model_name and model_type:
        # step2: load converted inference model
        tlx_model = load_inference_model_tlx(project_dst_path, model_name, model_type)
        # tlx_model, pd_model = load_inference_model(project_src_path, project_dst_path, model_name, model_type)

        # step3: validate the inference model of converted project
        validation_tlx(tlx_model, model_name, model_type)
        # validation(tlx_model, pd_model, model_name, model_type)
    # print('Done!')
