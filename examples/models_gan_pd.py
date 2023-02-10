# coding: utf-8
import os
import sys


class PaddleGenerateModel(object):
    def __init__(self, project_path, model_name="cyclegan"):
        self.model_name = model_name
        self.pd_model = self.load_pd_model(project_path, model_name)

    def load_pd_model(self, pd_project_path, model_name="cyclegan"):
        sys.path.insert(0, pd_project_path)

        model = None
        if model_name == "cyclegan":
            from models import cyclegan_model
            model = cyclegan_model._cyclegan(pretrained=True)
        elif model_name == "prenet":
            from models import prenet_model
            # model = prenet_model._prenet(pretrained=True)
            model = prenet_model._prenet(pretrained=f"{pd_project_path}/pretrain/PReNet.pdparams")
        elif model_name == "ugatit":
            from models import ugatit_model
            model = ugatit_model._ugatit(pretrained=True)
        elif model_name == "stylegan":
            from models import styleganv2_model
            # model = styleganv2_model._stylegan(pretrained=True)
            model = styleganv2_model._stylegan(pretrained=f"{pd_project_path}/pretrain/stylegan_v2_256_ffhq.pdparams")
        elif model_name == "stargan":
            from models import starganv2_model
            model = starganv2_model._stargan(pretrained=True)
        return model
