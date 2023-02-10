# coding: utf-8
import sys
import os
os.environ["TL_BACKEND"] = "paddle"
import importlib


class TLXSegmentationModel(object):
    def __init__(self, project_path, model_name="fast_scnn"):
        self.model_name = model_name
        self.tlx_model = self.load_tlx_model(project_path, model_name)

    def load_tlx_model(self, tlx_project_path, model_name="fast_scnn"):
        sys.path.insert(0, tlx_project_path)
        if model_name in ["fastfcn", "fast_scnn", "enet", "hrnet", "encnet", "bisenet"]:
            from models.load_test_config import TestConfig
            model_dict = {
                'fastfcn': f'{tlx_project_path}/configs/fastfcn/fastfcn_resnet50_os8_ade20k_480x480_120k.yml',
                'fast_scnn': f'{tlx_project_path}/configs/fastscnn/fastscnn_cityscapes_1024x1024_160k.yml',
                'enet': f'{tlx_project_path}/configs/enet/enet_cityscapes_1024x512_80k.yml',
                'hrnet': f'{tlx_project_path}/configs/hrnet_w48_contrast/HRNet_W48_contrast_cityscapes_1024x512_60k.yml',
                'encnet': f'{tlx_project_path}/configs/encnet/encnet_resnet101_os8_cityscapes_1024x512_80k.yml',
                'bisenet': f'{tlx_project_path}/configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml'
            }
            arg_cfg = model_dict[model_name]
            cfg = TestConfig(arg_cfg)
            model = cfg.model
            return model
        elif model_name in ["unet", "farseg", "deeplabv3p"]:
            model = None
            if model_name == "unet":
                from models import unet
                model = unet._unet(pretrained=True)
            elif model_name == "farseg":
                from models import farseg
                # model = farseg._farseg(pretrained=True)
                model = farseg._farseg(pretrained=f"{tlx_project_path}/pretrain/farseg/model.pdparams")
            elif model_name == "deeplabv3p":
                from models import deeplab
                model = deeplab._deeplabv3p(pretrained=True)
            return model
        else:
            raise ValueError(f"{model_name} is not currently supported.")
