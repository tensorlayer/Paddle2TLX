# coding: utf-8
import os
os.environ["TL_BACKEND"] = "paddle"
import sys
import configparser
import paddle

tlx_project_path = '/home/sthq/scc/paddle2tlx/my_project/paddlers/paddledetection'
model_name = 'YOLOX'


def load_tlx_model(tlx_project_path, model_name="YOLOV3"):
    sys.path.insert(0, tlx_project_path)
    os.chdir(tlx_project_path)
    from tools.train_det import parse_args
    from det.core.workspace import load_config
    from det.core.workspace import merge_config
    from det.core.workspace import merge_args
    from det.engine.trainer import Trainer
    from det.utils.check import check_gpu
    from det.utils.check import check_version
    from det.utils.check import check_config

    config = configparser.ConfigParser()
    config.read('/home/sthq/scc/paddle2tlx/my_project/translated_models.cfg')
    config_file = config.get('MODEL_CONFIG_PATH', model_name)
    print(config_file, '----------------')
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
    # results = run(FLAGS, cfg)
    trainer = Trainer(cfg=cfg, mode='train')
    return trainer

tlx_model = load_tlx_model(tlx_project_path, model_name)
tlx_model.train()