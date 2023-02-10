# coding: utf-8
import argparse
from six import text_type as _text_type
from paddle2tlx.common.convert import main as convert_paddle


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        "-mn",
        type=_text_type,
        default="",
        help="model name for paddle project"
    )
    parser.add_argument(
        "--model_type",
        "-mt",
        type=_text_type,
        default="",
        help="model type - clas|seg|det|cdet|gan|nlp"
    )
    parser.add_argument(
        "--input_dir_pd",
        "-idp",
        type=_text_type,
        # default="D:/ProjectByPython/code/myproject/model-convert-tools/convert_test/paddleclas",
        # default="../pd_models/paddlerscd",
        # default="../pd_models/paddleseg",
        default="../pd_models/test",
        help="define project folder path for paddle")
    parser.add_argument(
        "--output_dir_tlx",
        "-odt",
        type=_text_type,
        # default="D:/ProjectByPython/code/myproject/model-convert-tools/convert_test/paddleclas",
        # default="../tlx_models/paddlerscd",
        # default="../tlx_models/paddleseg",
        default="../tlx_models/test",
        help="path to save the converted model of tensorlayerx")
    parser.add_argument(
        "--save_tag",
        "-st",
        type=bool,
        default=True,
        help="save the converted model or not"
    )
    parser.add_argument(
        "--pretrain_model",
        "-pm",
        type=_text_type,
        default="./pretrain",
        help="pretrain model save path of converted model")
    args = parser.parse_args()
    return args


def main():
    args = get_params()
    convert_paddle(args)
    print('Done!')


if __name__ == "__main__":
    main()
