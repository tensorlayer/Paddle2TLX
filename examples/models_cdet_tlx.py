# coding: utf-8
import os
os.environ["TL_BACKEND"] = "paddle"
import sys


class TLXChangeDetectionModel(object):
    def __init__(self, project_path, model_name="bit"):
        self.model_name = model_name
        self.tlx_model = self.load_tlx_model(project_path, model_name)

    def load_tlx_model(self, tlx_project_path, model_name="bit"):
        sys.path.insert(0, tlx_project_path)
        # os.chdir(tlx_project_path)

        model = None
        if model_name == "bit":  # ok
            from models import bit
            model = bit._bit(pretrained=True)
        elif model_name == "cdnet":  # ok
            from models import cdnet
            model = cdnet._cdnet(pretrained=True)
        elif model_name == "stanet":  # ok
            from models import stanet
            model = stanet._stanet(pretrained=True)
        elif model_name == "fcef":  # has diff
            from models import fc_ef
            model = fc_ef._fcef(pretrained=True)
        elif model_name == "fccdn":  # ok
            from models import fccdn
            model = fccdn._fccdn(pretrained=True)
        elif model_name == "dsamnet":  # ok
            from models import dsamnet
            model = dsamnet._dsamnet(pretrained=True)
        elif model_name == "snunet":  # ok
            from models import snunet
            model = snunet._snunet(pretrained=True)
        elif model_name == "dsifn":
            from models import dsifn
            model = dsifn._dsifn(pretrained=True)
        return model
