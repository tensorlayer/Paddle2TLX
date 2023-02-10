from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tqdm import tqdm

import typing
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import paddle.nn as nn

from core.workspace import create
from utils.checkpoint import load_pretrain_weight

from utils.logger import setup_logger
logger = setup_logger('det.engine')

__all__ = ['predicter']


class predicter(object):
    def __init__(self, cfg, mode='test'):
        self.cfg = cfg
        assert mode.lower() in ['train', 'test'], \
                "mode should be 'train', or 'test'"
        self.mode = mode.lower()
        self.optimizer = None
        self.is_loaded_weights = False
        self.custom_white_list = self.cfg.get('custom_white_list', None)
        self.custom_black_list = self.cfg.get('custom_black_list', None)

        # build data loader
        capital_mode = self.mode.capitalize()
        self.dataset = self.cfg['{}Dataset'.format(capital_mode)] = create(
                '{}Dataset'.format(capital_mode))()

        # build model
        if 'model' not in self.cfg:
            self.model = create(cfg.architecture)
        else:
            self.model = self.cfg.model
            self.is_loaded_weights = True

        self.model.load_meanstd(cfg['TestReader']['sample_transforms'])

        self.status = {}

        self.start_epoch = 0
        self.end_epoch = 0 if 'epoch' not in cfg else cfg.epoch

    def load_weights(self, weights):
        if self.is_loaded_weights:
            return
        self.start_epoch = 0
        load_pretrain_weight(self.model, weights)
        logger.debug("Load weights {} to start training".format(weights))

    def predict(self,
                images,
                output_dir='output',):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        # Run Infer 
        self.status['mode'] = 'test'
        self.model.eval()
        results = []
        print(f"predict ... images={images}")
        for step_id, data in enumerate(tqdm(loader)):
            # print(f"predict ... data.keys()={data.keys()}")
            self.status['step_id'] = step_id

            # outs = self.model(data)
            outs = self.model(data["image"])

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            results.append(outs)
        return results
                    
    def _get_save_image_name(self, output_dir, image_path):
        """
        Get save image name from source image path.
        """
        image_name = os.path.split(image_path)[-1]
        name, ext = os.path.splitext(image_name)
        return os.path.join(output_dir, "{}".format(name)) + ext
