# coding: utf-8
import os
import sys


class PaddleChangeDetectionModel(object):
    def __init__(self, project_path, model_name="bit"):
        self.model_name = model_name
        self.pd_model = self.load_pd_model(project_path, model_name)

    def load_pd_model(self, pd_project_path, model_name="bit"):
        sys.path.insert(0, pd_project_path)
        # os.chdir(pd_project_path)

        model = None
        if model_name == "bit":
            from models import bit
            model = bit._bit(pretrained=True)
        elif model_name == "cdnet":
            from models import cdnet
            model = cdnet._cdnet(pretrained=True)
        elif model_name == "stanet":
            from models import stanet
            model = stanet._stanet(pretrained=True)
        elif model_name == "fcef":
            from models import fc_ef
            model = fc_ef._fcef(pretrained=True)
        elif model_name == "fccdn":
            from models import fccdn
            model = fccdn._fccdn(pretrained=True)
        elif model_name == "dsamnet":
            from models import dsamnet
            model = dsamnet._dsamnet(pretrained=True)
        elif model_name == "snunet":
            from models import snunet
            model = snunet._snunet(pretrained=True)
        elif model_name == "dsifn":
            from models import dsifn
            model = dsifn._dsifn(pretrained=True)
        return model
