from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import time
from tqdm import tqdm

import numpy as np
import typing
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import paddle
import paddle.nn as nn

from core.workspace import create
from utils.checkpoint import load_pretrain_weight
# from utils.visualizer import visualize_results
# from metrics import get_infer_results
# from data.source.category import get_categories
import utils.stats as stats

from .callbacks import Callback, ComposeCallback, LogPrinter, Checkpointer


from utils.logger import setup_logger
logger = setup_logger('det.engine')

__all__ = ['Trainer']


class Trainer(object):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        # print(f"self.cfg={self.cfg}")
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

        if self.mode == 'train':
            # print(f"capital_mode={capital_mode}")
            self.loader = create('{}Reader'.format(capital_mode))(
                self.dataset, cfg.worker_num)

        # build model
        if 'model' not in self.cfg:
            self.model = create(cfg.architecture)
        else:
            self.model = self.cfg.model
            self.is_loaded_weights = True

        # if cfg.architecture == 'YOLOX':
        #     for k, m in self.model.named_sublayers(): 
        #         if isinstance(m, nn.BatchNorm2D):
        #             m._epsilon = 1e-3  # for amp(fp16)
        #             m._momentum = 0.97  # 0.03 in pytorch

        self.model.load_meanstd(cfg['TestReader']['sample_transforms'])

        # build optimizer in train mode
        if self.mode == 'train':
            steps_per_epoch = len(self.loader)
            if steps_per_epoch < 1:
                logger.warning(
                    "Samples in dataset are less than batch_size, please set smaller batch_size in TrainReader."
                )
            self.lr = self.cfg['LearningRate']['base_lr']
            # print(f"self.lr={self.lr}")
            # print(f"self.model={self.model}")
            self.optimizer = create('OptimizerBuilder')(self.lr, self.model)
            # self.optimizer = create('OptimizerBuilder')

            # Unstructured pruner is only enabled in the train mode.
            if self.cfg.get('unstructured_prune'):
                self.pruner = create('UnstructuredPruner')(self.model,
                                                           steps_per_epoch)

        self.status = {}

        self.start_epoch = 0
        self.end_epoch = 0 if 'epoch' not in cfg else cfg.epoch

        self._init_callbacks()

    def _init_callbacks(self):
        if self.mode == 'train': 
            self._callbacks = [LogPrinter(self), Checkpointer(self)]
            self._compose_callback = ComposeCallback(self._callbacks)
        elif self.mode == 'test' and self.cfg.get('use_vdl', False):
            self._compose_callback = ComposeCallback(self._callbacks)
        else:
            self._callbacks = []
            self._compose_callback = None

    def _set_eval_pd(self):
        self.model.eval()

    def _set_eval_tlx(self):
        self.model.set_eval()
        self.model.eval()

    def register_callbacks(self, callbacks):
        callbacks = [c for c in list(callbacks) if c is not None]
        for c in callbacks:
            assert isinstance(c, Callback), \
                    "metrics shoule be instances of subclass of Metric"
        self._callbacks.extend(callbacks)
        self._compose_callback = ComposeCallback(self._callbacks)

    def load_weights(self, weights):
        if self.is_loaded_weights:
            return
        self.start_epoch = 0
        load_pretrain_weight(self.model, weights)
        logger.debug("Load weights {} to start training".format(weights))

    def train(self):
        assert self.mode == 'train', "Model not in 'train' mode"

        model = self.model

        self.status.update({
            'epoch_id': self.start_epoch,
            'step_id': 0,
            'steps_per_epoch': len(self.loader)
        })

        self.status['batch_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['data_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['training_staus'] = stats.TrainingStats(self.cfg.log_iter)

        if self.cfg.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg.worker_num)
            self._flops(flops_loader)

        self._compose_callback.on_train_begin(self.status)

        for epoch_id in range(self.start_epoch, self.cfg.epoch):
            self.status['mode'] = 'train'
            self.status['epoch_id'] = epoch_id
            self._compose_callback.on_epoch_begin(self.status)
            self.loader.dataset.set_epoch(epoch_id)
            model.train()
            iter_tic = time.time()
            for step_id, data in enumerate(self.loader):
                if step_id > 500:
                    break
                self.status['data_time'].update(time.time() - iter_tic)
                self.status['step_id'] = step_id
                self._compose_callback.on_step_begin(self.status)
                data['epoch_id'] = epoch_id

                # model forward
                outputs = model(data)
                loss = outputs['loss']
                # model backward
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
                
                self.status['training_staus'].update(outputs)
                self.status['batch_time'].update(time.time() - iter_tic)
                self._compose_callback.on_step_end(self.status)
                
                iter_tic = time.time()

            self._compose_callback.on_epoch_end(self.status)


        self._compose_callback.on_train_end(self.status)


    def predict(self,
                images,
                output_dir='output',):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        # Run Infer 
        self.status['mode'] = 'test'
        # self.model.eval()  # todo
        # self._set_eval()
        self._set_eval_pd()
        results = []
        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            outs = self.model(data)

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
                    
    # def get_predict_results(self, images):
    #     self.dataset.set_images(images)
    #     loader = create('TestReader')(self.dataset, 0)

    #     # Run Infer 
    #     self.status['mode'] = 'test'
    #     self.model.set_eval()

    #     results = []
    #     for step_id, data in enumerate(tqdm(loader)):
    #         self.status['step_id'] = step_id
    #         # forward
    #         outs = self.model(data)
    #         results.append(outs)
            
    #     return results

    def _get_save_image_name(self, output_dir, image_path):
        """
        Get save image name from source image path.
        """
        image_name = os.path.split(image_path)[-1]
        name, ext = os.path.splitext(image_name)
        return os.path.join(output_dir, "{}".format(name)) + ext
