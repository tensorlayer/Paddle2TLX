from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import os
import datetime
from utils.logger import setup_logger
logger = setup_logger('det.engine')
__all__ = ['Callback', 'ComposeCallback', 'LogPrinter', 'Checkpointer']


class Callback(object):

    def __init__(self, model):
        self.model = model

    def on_step_begin(self, status):
        pass

    def on_step_end(self, status):
        pass

    def on_epoch_begin(self, status):
        pass

    def on_epoch_end(self, status):
        pass

    def on_train_begin(self, status):
        pass

    def on_train_end(self, status):
        pass


class ComposeCallback(object):

    def __init__(self, callbacks):
        callbacks = [c for c in list(callbacks) if c is not None]
        for c in callbacks:
            assert isinstance(c, Callback
                ), 'callback should be subclass of Callback'
        self._callbacks = callbacks

    def on_step_begin(self, status):
        for c in self._callbacks:
            c.on_step_begin(status)

    def on_step_end(self, status):
        for c in self._callbacks:
            c.on_step_end(status)

    def on_epoch_begin(self, status):
        for c in self._callbacks:
            c.on_epoch_begin(status)

    def on_epoch_end(self, status):
        for c in self._callbacks:
            c.on_epoch_end(status)

    def on_train_begin(self, status):
        for c in self._callbacks:
            c.on_train_begin(status)

    def on_train_end(self, status):
        for c in self._callbacks:
            c.on_train_end(status)


class LogPrinter(Callback):

    def __init__(self, model):
        super(LogPrinter, self).__init__(model)

    def on_step_end(self, status):
        mode = status['mode']
        if mode == 'train':
            epoch_id = status['epoch_id']
            step_id = status['step_id']
            steps_per_epoch = status['steps_per_epoch']
            training_staus = status['training_staus']
            batch_time = status['batch_time']
            data_time = status['data_time']
            epoches = self.model.cfg.epoch
            batch_size = self.model.cfg['{}Reader'.format(mode.capitalize())][
                'batch_size']
            logs = training_staus.log()
            space_fmt = ':' + str(len(str(steps_per_epoch))) + 'd'
            if step_id % self.model.cfg.log_iter == 0:
                eta_steps = (epoches - epoch_id) * steps_per_epoch - step_id
                eta_sec = eta_steps * batch_time.global_avg
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                ips = float(batch_size) / batch_time.avg
                fmt = ' '.join(['Epoch: [{}]', '[{' + space_fmt + '}/{}]'])
                fmt = fmt.format(epoch_id, step_id, steps_per_epoch)
                logger.info(fmt)
        if mode == 'eval':
            step_id = status['step_id']
            if step_id % 100 == 0:
                logger.info('Eval iter: {}'.format(step_id))

    def on_epoch_end(self, status):
        mode = status['mode']
        if mode == 'eval':
            sample_num = status['sample_num']
            cost_time = status['cost_time']
            logger.info('Total sample number: {}, averge FPS: {}'.format(
                sample_num, sample_num / cost_time))


class Checkpointer(Callback):

    def __init__(self, model):
        super(Checkpointer, self).__init__(model)
        cfg = self.model.cfg
        self.best_ap = 0.0
        self.save_dir = os.path.join(self.model.cfg.save_dir, self.model.
            cfg.filename)
        if hasattr(self.model.model, 'student_model'):
            self.weight = self.model.model.student_model
        else:
            self.weight = self.model.model

    def on_epoch_end(self, status):
        mode = status['mode']
        epoch_id = status['epoch_id']
        if mode == 'eval':
            for metric in self.model._metrics:
                map_res = metric.get_results()
                if 'bbox' in map_res:
                    key = 'bbox'
                elif 'keypoint' in map_res:
                    key = 'keypoint'
                else:
                    key = 'mask'
                if key not in map_res:
                    logger.warning(
                        'Evaluation results empty, this may be due to training iterations being too few or not loading the correct weights.'
                        )
                    return
                if map_res[key][0] >= self.best_ap:
                    self.best_ap = map_res[key][0]
                logger.info('Best test {} ap is {:0.3f}.'.format(key, self.
                    best_ap))
