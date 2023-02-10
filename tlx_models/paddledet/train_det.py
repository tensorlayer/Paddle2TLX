from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TL_BACKEND'] = 'paddle'
import tensorlayerx as tlx
import paddle
import paddle2tlx
import os
os.environ['TL_BACKEND'] = 'paddle'
import sys
import argparse
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)
import warnings
warnings.filterwarnings('ignore')
import tensorlayerx
from core.workspace import load_config
from core.workspace import merge_config
from core.workspace import merge_args
from engine import Trainer
import utils.check as check
from utils.logger import setup_logger
logger = setup_logger('train')
import configparser


def parse_args(config_file=None, weights_file=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file, help=\
        'config path for yaml file')
    parser.add_argument('--use_gpu', type=bool, default=True, help='')
    parser.add_argument('-o', '--opt', nargs='*', default={'use_gpu': True},
        help='set configuration options')
    parser.add_argument('--pretrain_weights', type=str, default=\
        weights_file, help='pretrained weights path or url')
    parser.add_argument('-r', '--resume', default=None, help=\
        'weights path for resume')
    parser.add_argument('--save_prediction_only', action='store_true',
        default=False, help='Whether to save the evaluation results only')
    parser.add_argument('--profiler_options', type=str, default=None, help=\
        'The option of profiler, which should be in format "key1=value1;key2=value2;key3=value3".please see det/utils/profiler.py for detail.'
        )
    parser.add_argument('--save_proposals', action='store_true', default=\
        False, help='Whether to save the train proposals')
    parser.add_argument('--proposals_path', type=str, default=\
        'sniper/proposals.json', help='Train proposals directory')
    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    trainer = Trainer(cfg, mode='train')
    if FLAGS.resume is not None:
        trainer.resume_weights(FLAGS.resume)
    elif 'pretrain_weights' in cfg and cfg.pretrain_weights:
        trainer.load_weights(FLAGS.pretrain_weights)
    trainer.train()


def main(model_name):
    config = configparser.ConfigParser()
    config.read('translated_models.cfg')
    config_file = config.get('MODEL_CONFIG_PATH', model_name)
    weights_file = config.get('MODEL_WEIGHTS_PATH', model_name)
    FLAGS = parse_args(config_file, weights_file)
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)
    if cfg.use_gpu:
        place = tensorlayerx.set_device('gpu')
    else:
        place = tensorlayerx.set_device('cpu')
    print(f'place={place}')
    merge_config(FLAGS.opt)
    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)
    check.check_version()
    run(FLAGS, cfg)


if __name__ == '__main__':
    model_name = 'faster_rcnn'
    main(model_name)
