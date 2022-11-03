# coding: utf-8
import sys


class PaddleClassificationModel(object):
    def __init__(self, project_path, model_name="vgg16"):
        self.model_name = model_name
        self.pd_model = self.load_pd_model(project_path, model_name)

    def load_pd_model(self, pd_project_path, model_name="vgg16"):
        sys.path.insert(0, pd_project_path)
        model = None
        if model_name == "vgg16":
            from vgg import vgg16 as pd_vgg16
            model = pd_vgg16(pretrained=True)
        elif model_name == "alexnet":
            from alexnet import alexnet as pd_alexnet
            model = pd_alexnet(pretrained=True)
        elif model_name == "resnet50":
            from resnet import resnet50 as pd_resnet50
            model = pd_resnet50(pretrained=True)
        elif model_name == "resnet101":
            from resnet import resnet101 as pd_resnet101
            model = pd_resnet101(pretrained=True)
        elif model_name == "googlenet":
            from googlenet import googlenet as pd_googlenet
            model = pd_googlenet(pretrained=True)
        elif model_name == "mobilenetv1":
            from mobilenetv1 import mobilenet_v1 as pd_mobilenet_v1
            model = pd_mobilenet_v1(pretrained=True)
        elif model_name == "mobilenetv2":
            from mobilenetv2 import mobilenet_v2 as pd_mobilenet_v2
            model = pd_mobilenet_v2(pretrained=True)
        elif model_name == "mobilenetv3":
            from mobilenetv3 import mobilenet_v3_small as pd_mobilenet_v3_small
            from mobilenetv3 import mobilenet_v3_large as pd_mobilenet_v3_large
            model = pd_mobilenet_v3_small(pretrained=True)
            # model = pd_mobilenet_v3_large(pretrained=True)
        elif model_name == "shufflenetv2":
            from shufflenetv2 import shufflenet_v2_swish as pd_shufflenetv2
            model = pd_shufflenetv2(pretrained=True)
        elif model_name == "squeezenet":
            from squeezenet import squeezenet1_0 as pd_squeezenet1_0
            # from squeezenet import squeezenet1_1 as pd_squeezenet1_1
            model = pd_squeezenet1_0(pretrained=True)
        elif model_name == "inceptionv3":
            from inceptionv3 import inception_v3 as pd_inception_v3
            model = pd_inception_v3(pretrained=True)
        elif model_name == "regnet":
            from regnet import RegNetX_4GF as pd_RegNetX_4GF
            model = pd_RegNetX_4GF()
        elif model_name == "tnt":
            from tnt import TNT_small as pd_TNT_small
            model = pd_TNT_small(pretrained=True)
        sys.path.remove(pd_project_path)
        return model
