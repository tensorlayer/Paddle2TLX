# coding: utf-8
import sys


class TLXClassificationModel(object):
    def __init__(self, project_path, model_name="vgg16"):
        self.model_name = model_name
        self.tlx_model = self.load_tlx_model(project_path, model_name)

    def load_tlx_model(self, tlx_project_path, model_name="vgg16"):
        import os
        os.environ["TL_BACKEND"] = "paddle"
        sys.path.insert(0, tlx_project_path)
        model = None
        if model_name == "vgg16":
            from vgg import vgg16 as tlx_vgg16
            model = tlx_vgg16(pretrained=True)
        elif model_name == "alexnet":
            from alexnet import alexnet as tlx_alexnet
            model = tlx_alexnet(pretrained=True)
        elif model_name == "resnet50":
            from resnet import resnet50 as tlx_resnet50
            model = tlx_resnet50(pretrained=True)
        elif model_name == "resnet101":
            from resnet import resnet101 as tlx_resnet101
            model = tlx_resnet101(pretrained=True)
        elif model_name == "googlenet":
            from googlenet import googlenet as tlx_googlenet
            model = tlx_googlenet(pretrained=True)
        elif model_name == "mobilenetv1":
            from mobilenetv1 import mobilenet_v1 as tlx_mobilenet_v1
            model = tlx_mobilenet_v1(pretrained=True)
        elif model_name == "mobilenetv2":
            from mobilenetv2 import mobilenet_v2 as tlx_mobilenet_v2
            model = tlx_mobilenet_v2(pretrained=True)
        elif model_name == "mobilenetv3":
            from mobilenetv3 import mobilenet_v3_small as tlx_mobilenet_v3_small
            from mobilenetv3 import mobilenet_v3_large as tlx_mobilenet_v3_large
            model = tlx_mobilenet_v3_small(pretrained=True)
            # model = tlx_mobilenet_v3_large(pretrained=True)
        elif model_name == "shufflenetv2":
            from shufflenetv2 import shufflenet_v2_swish as tlx_shufflenetv2
            model = tlx_shufflenetv2(pretrained=True)
        elif model_name == "squeezenet":
            from squeezenet import squeezenet1_0 as tlx_squeezenet1_0
            # from squeezenet import squeezenet1_1 as tlx_squeezenet1_1
            model = tlx_squeezenet1_0(pretrained=True)
        elif model_name == "inceptionv3":
            from inceptionv3 import inception_v3 as tlx_inception_v3
            model = tlx_inception_v3(pretrained=True)
        elif model_name == "regnet":
            from regnet import RegNetX_4GF as tlx_RegNetX_4GF
            model = tlx_RegNetX_4GF(pretrained=True)
        elif model_name == "tnt":
            from tnt import TNT_small as tlx_TNT_small
            model = tlx_TNT_small(pretrained=True)
        sys.path.remove(tlx_project_path)
        return model
