from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TL_BACKEND'] = 'paddle'
import tensorlayerx as tlx
import paddle
import paddle2tlx
import os
import sys
import argparse
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)
import warnings
warnings.filterwarnings('ignore')
import glob
import configparser
import tensorlayerx
from core.workspace import load_config
from core.workspace import merge_config
from core.workspace import merge_args
from engine import Trainer
from engine import predicter


def parse_args(config_file=None, weights_file=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=config_file,
        help='config path for yaml file')
    parser.add_argument('--use_gpu', type=bool, default=False, help='')
    parser.add_argument('-o', '--opt', nargs='*', default={'use_gpu': False
        }, help='set configuration options')
    parser.add_argument('--pretrain_weights', type=str, default=\
        weights_file, help='pretrained weights path or url')
    parser.add_argument('--infer_dir', type=str, default=None, help=\
        'Directory for images to perform inference on.')
    parser.add_argument('--infer_img', type=str, default=\
        'demo/000000014439.jpg', help=\
        'Image path, has higher priority over --infer_dir')
    parser.add_argument('--output_dir', type=str, default='output', help=\
        'Directory for storing the output visualization files.')
    args = parser.parse_args()
    return args


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, '--infer_img or --infer_dir should be set'
    assert infer_img is None or os.path.isfile(infer_img
        ), '{} is not a file'.format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir
        ), '{} is not a directory'.format(infer_dir)
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]
    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), 'infer_dir {} is not a directory'.format(
        infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)
    assert len(images) > 0, 'no image found in {}'.format(infer_dir)
    print('Found {} inference images in total.'.format(len(images)))
    return images


def run(FLAGS, cfg):
    trainer = Trainer(cfg=cfg, mode='test')
    trainer.load_weights(FLAGS.pretrain_weights)
    images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
    results = trainer.predict(images, output_dir=FLAGS.output_dir)
    return results


def main(model_name):
    config = configparser.ConfigParser()
    config.read('translated_models.cfg')
    config_file = config.get('MODEL_CONFIG_PATH', model_name)
    weights_file = config.get('MODEL_WEIGHTS_PATH', model_name)
    FLAGS = parse_args(config_file, weights_file)
    print(f'FLAGS.config={FLAGS.config}')
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)
    if FLAGS.use_gpu:
        place = tensorlayerx.set_device('gpu')
    else:
        place = tensorlayerx.set_device('cpu')
    results = run(FLAGS, cfg)
    print(results)
    return results


if __name__ == '__main__':
    model_name = 'YOLOX'
    img = main(model_name)
