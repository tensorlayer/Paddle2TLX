# coding: utf-8
import argparse


def load_inference_model_pd(pd_project_path, model_name="vgg16", model_type="clas"):
    if model_type == "clas":
        from examples.models_clas_pd import PaddleClassificationModel
        ModelPaddle = PaddleClassificationModel(pd_project_path, model_name)
        pd_model = ModelPaddle.pd_model
        return pd_model
    elif model_type == "seg":
        from examples.models_seg_pd import PaddleSegmentationModel
        ModelPaddle = PaddleSegmentationModel(pd_project_path, model_name)
        pd_model = ModelPaddle.pd_model
        return pd_model
    elif model_type == "det":
        from examples.models_det_pd import PaddleDetectionModel
        ModelPaddle = PaddleDetectionModel(pd_project_path, model_name)
        pd_model = ModelPaddle.pd_model
        return pd_model
    elif model_type == "cdet":
        from examples.models_cdet_pd import PaddleChangeDetectionModel
        ModelPaddle = PaddleChangeDetectionModel(pd_project_path, model_name)
        pd_model = ModelPaddle.pd_model
        return pd_model
    elif model_type == "nlp":
        from examples.models_nlp_pd import PaddleClassificationModel
        ModelPaddle = PaddleClassificationModel(pd_project_path, model_name)
        pd_model = ModelPaddle.pd_model
        return pd_model
    elif model_type == "gan":
        from examples.models_gan_pd import PaddleGenerateModel
        ModelPaddle = PaddleGenerateModel(pd_project_path, model_name)
        pd_model = ModelPaddle.pd_model
        return pd_model


def validation_pd(pd_model, model_name="vgg16", model_type="clas"):
    if model_type == "clas":
        from examples.predict_clas import predict_pd
        predict_pd(pd_model, model_name)
    elif model_type == "seg":
        if model_name in ["fastfcn", "fast_scnn", "enet", "hrnet", "encnet", "bisenet"]:
            from examples.predict_seg import predict_pd
            predict_pd(pd_model, model_name)
        elif model_name in ["unet", "farseg", "deeplabv3p"]:
            from examples.predict_rsseg import predict_pd
            predict_pd(pd_model, model_name)
    elif model_type == "det":
        from examples.predict_det import predict_pd
        predict_pd(pd_model, model_name)
    elif model_type == "cdet":
        from examples.predict_cdet import predict_pd
        predict_pd(pd_model, model_name)
    elif model_type == "nlp":
        from examples.predict_nlp import predict_pd
        predict_pd(pd_model, model_name)
    elif model_type == "gan":
        from examples.predict_gan import predict_pd
        predict_pd(pd_model, model_name)


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
    elif model_type == "gan":
        from examples.models_gan_tlx import TLXGenerateModel
        ModelTLX = TLXGenerateModel(tlx_project_path, model_name)
        tlx_model = ModelTLX.tlx_model
        return tlx_model


def validation_tlx(tlx_model, model_name="vgg16", model_type="clas"):
    if model_type == "clas":
        from examples.predict_clas import predict_tlx
        predict_tlx(tlx_model, model_name)
    elif model_type == "seg":
        if model_name in ["fastfcn", "fast_scnn", "enet", "hrnet", "encnet", "bisenet"]:
            from examples.predict_seg import predict_tlx
            predict_tlx(tlx_model, model_name)
        elif model_name in ["unet", "farseg", "deeplabv3p"]:
            from examples.predict_rsseg import predict_tlx
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
    elif model_type == "gan":
        from examples.predict_gan import predict_tlx
        predict_tlx(tlx_model, model_name)


# def load_inference_model(pd_project_path, tlx_project_path, model_name="vgg16", model_type="clas"):
#     if model_type == "clas":
#         from examples.models_clas_tlx import TLXClassificationModel
#         from examples.models_clas_pd import PaddleClassificationModel
#         ModelTLX = TLXClassificationModel(tlx_project_path, model_name)  # load in first
#         tlx_model = ModelTLX.tlx_model
#         ModelPaddle = PaddleClassificationModel(pd_project_path, model_name)
#         pd_model = ModelPaddle.pd_model
#         return tlx_model, pd_model
#     elif model_type == "seg":
#         from examples.models_seg_tlx import TLXSegmentationModel
#         from examples.models_seg_pd import PaddleSegmentationModel
#         ModelTLX = TLXSegmentationModel(tlx_project_path, model_name)  # load in first
#         tlx_model = ModelTLX.tlx_model
#         ModelPaddle = PaddleSegmentationModel(pd_project_path, model_name)
#         pd_model = ModelPaddle.pd_model
#         return tlx_model, pd_model
#     elif model_type == "det":
#         from examples.models_det_tlx import TLXDetectionModel
#         from examples.models_det_pd import PaddleDetectionModel
#         ModelTLX = TLXDetectionModel(tlx_project_path, model_name)
#         tlx_model = ModelTLX.tlx_model
#         ModelPaddle = PaddleDetectionModel(tlx_project_path, model_name)
#         pd_model = ModelPaddle.pd_model
#         return tlx_model, pd_model
#     elif model_type == "cdet":
#         from examples.models_cdet_tlx import TLXChangeDetectionModel
#         from examples.models_cdet_pd import PaddleChangeDetectionModel
#         ModelTLX = TLXChangeDetectionModel(tlx_project_path, model_name)
#         tlx_model = ModelTLX.tlx_model
#         ModelPaddle = PaddleChangeDetectionModel(tlx_project_path, model_name)
#         pd_model = ModelPaddle.pd_model
#         return tlx_model, pd_model
#
#
# def validation(tlx_model, pd_model, model_name="vgg16", model_type="clas"):
#     """ validate model prediction result of converted tensorlayerx models """
#     if model_type == "clas":
#         from examples.predict_clas import calc_diff
#         from examples.predict_clas import predict_tlx, predict_pd
#         calc_diff(tlx_model, pd_model, model_name)
#         predict_tlx(tlx_model, model_name)
#         predict_pd(pd_model, model_name)
#     elif model_type == "seg":
#         from examples.predict_seg import predict_tlx, predict_pd, calc_diff
#         result_tlx = predict_tlx(tlx_model, model_name)
#         result_pd = predict_pd(pd_model, model_name)
#         calc_diff(result_tlx, result_pd, model_name)
#     elif model_type == "det":
#         from examples.predict_det import predict_tlx, predict_pd, calc_diff
#         result_tlx = predict_tlx(tlx_model, model_name)
#         result_pd = predict_pd(pd_model, model_name)
#         calc_diff(result_tlx, result_pd, model_name)
#     elif model_type == "cdet":
#         from examples.predict_cdet import predict_tlx, predict_pd, calc_diff
#         result_tlx = predict_tlx(tlx_model, model_name)
#         result_pd = predict_pd(pd_model, model_name)
#         calc_diff(result_tlx, result_pd, model_name)


def parse_args():
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument("--model_name", default="vgg16", help="model name of project")
    parser.add_argument("--model_type", default="clas", help="model name of project")
    parser.add_argument("--project_type",
                        default="tlx",
                        help="paddle model before conversion or tlx model after conversion")
    parser.add_argument(
        "--input_dir_pd",
        default="../pd_models/paddleclas",
        # default="../pd_models/paddlenlp",
        # default="../pd_models/paddlerscd",
        # default="../pd_models/paddlersseg",
        # default="../pd_models/paddleseg",
        # default="../pd_models/paddlegan",
        # default="../pd_models/paddledet",
        help="define project folder path for paddle")
    parser.add_argument(
        "--output_dir_tlx",
        default="../tlx_models/paddleclas",   # ok
        # default="../tlx_models/paddlenlp",    # ok
        # default="../tlx_models/paddlerscd",   # ok
        # default="../tlx_models/paddlersseg",  # ok
        # default="../tlx_models/paddleseg",    # ok - run predict_gan.py in project dir
        # default="../tlx_models/paddlegan",    # ok - run infer_det.py in project dir
        # default="../tlx_models/paddledet",    # ok
        help="path to save the converted model of tensorlayerx")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    project_src_path = args.input_dir_pd
    project_dst_path = args.output_dir_tlx
    model_name = args.model_name
    model_type = args.model_type
    project_type = args.project_type

    if project_type == "pd":
        pd_model = load_inference_model_pd(project_src_path, model_name, model_type)
        validation_pd(pd_model, model_name, model_type=model_type)
    elif project_type == "tlx":
        tlx_model = load_inference_model_tlx(project_dst_path, model_name, model_type)
        validation_tlx(tlx_model, model_name, model_type=model_type)
    # tlx_model, pd_model = load_inference_model(project_src_path,
    #                                            project_dst_path,
    #                                            model_name,
    #                                            model_type="nlp")
    # validation(tlx_model, pd_model, model_name, model_type="nlp")
    print('Done!')


if __name__ == '__main__':
    main()
