# coding: utf-8
import sys


class ClassificationModel(object):
    def __init__(self, project_src_path, project_dst_path, model_name="vgg16"):
        self.model_name = model_name
        self.pd_model = self.load_pd_model(project_src_path, model_name)
        self.tlx_model = self.load_tlx_model(project_dst_path, model_name)

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
        sys.path.remove(tlx_project_path)
        return model

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
        sys.path.remove(pd_project_path)
        return model


if __name__ == '__main__':
    pd_project_path = "D:/ProjectByPython/code/myproject/model-convert-tools/convert_test/pd_models"
    tlx_project_path = "D:/ProjectByPython/code/myproject/model-convert-tools/convert_test/tlx_models"
    model_name = "vgg16"
    # ModelPaddle = ClassificationModel(pd_project_path, model_name)
    ModelTLX = ClassificationModel(tlx_project_path, model_name)
    # print(ModelPaddle.pd_model)
    print(ModelTLX.tlx_model)
