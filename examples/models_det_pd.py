# coding: utf-8
import sys
import os
import configparser
import numpy as np
import paddle


class PaddleDetectionModel(object):
    def __init__(self, project_path, model_name="yolov3"):
        self.model_name = model_name
        self.pd_model = self.load_pd_model(project_path, model_name)

    def predict(self, pd_project_path, model_name="yolov3"):
        sys.path.insert(0, pd_project_path)
        from tools.infer_det import main as pd_main

        res = 'bbox' if model_name != 'SOLOv2' else 'segm'
        result_pd = pd_main(model_name)
        result_pd = np.array(result_pd[0][res])
        print('result_pd:\n', result_pd)
        return result_pd

    def load_pd_model(self, pd_project_path, model_name="yolov3"):
        sys.path.insert(0, pd_project_path)
        os.chdir(pd_project_path)
        from tools.infer_det import parse_args
        from det.core.workspace import load_config
        from det.core.workspace import merge_config
        from det.core.workspace import merge_args
        from det.engine import Trainer
        from det.utils.check import check_gpu
        from det.utils.check import check_version
        from det.utils.check import check_config

        config = configparser.ConfigParser()
        config.read('/home/sthq/scc/paddle2tlx/my_project/translated_models.cfg')
        config_file = config.get('MODEL_CONFIG_PATH', model_name)
        weights_file = config.get('MODEL_WEIGHTS_PATH', model_name)
        FLAGS = parse_args(config_file, weights_file)
        cfg = load_config(FLAGS.config)
        merge_args(cfg, FLAGS)
        merge_config(FLAGS.opt)
        if FLAGS.use_gpu:
            place = paddle.set_device('gpu')
        else:
            place = paddle.set_device('cpu')
        check_config(cfg)
        check_gpu(cfg.use_gpu)
        check_version()
        trainer = Trainer(cfg=cfg, mode='test')
        return trainer
