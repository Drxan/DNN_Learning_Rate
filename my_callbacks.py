# -*- coding: utf-8 -*-
# @Time    : 2018/9/29 17:25
# @Author  : Drxan
# @Email   : yuwei8905@126.com
# @File    : my_callbacks.py
# @Software: PyCharm
from keras.callbacks import *
import matplotlib.pyplot as plt
from IPython import get_ipython
from abc import abstractmethod


def in_ipynb():
    try:
        cls = get_ipython().__class__.__name__
        return cls == 'ZMQInteractiveShell'
    except NameError:
        return False


class CallbackT(Callback):
    def __init__(self):
        self.epoch_begin_logs = []
        self.epoch_end_logs = []
        self.train_begin_logs = []
        self.train_end_logs = []
        self.batch_begin_logs = []
        self.batch_end_logs = []
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_begin_logs.append([epoch, logs])

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_end_logs.append([epoch, logs])

    def on_batch_begin(self, batch, logs=None):
        self.batch_begin_logs.append([batch, logs])

    def on_batch_end(self, batch, logs=None):
        self.batch_end_logs.append([batch, logs])

    def on_train_begin(self, logs=None):
        self.train_begin_logs.append(logs)

    def on_train_end(self, logs=None):
        self.train_end_logs.append(logs)


class PerformanceLogger(Callback):
    """Callback that records events into a `PerformanceLogger` object at each batch.
    """
    def __init__(self):
        super(PerformanceLogger, self).__init__()
        self.batch_history = {}
        self.epoch_history = {}
        self.batch_num = -1

    def on_batch_end(self, batch, logs=None):
        self.batch_num += 1
        self.batch_history.setdefault('iteration', []).append(self.batch_num)
        logs = logs or {}
        for k, v in logs.items():
            self.batch_history.setdefault(k, []).append(v)

        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        self.batch_history.setdefault('lrs', []).append(lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.epoch_history.setdefault(k, []).append(v)

        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        self.epoch_history.setdefault('lrs', []).append(lr)

    def plot_loss(self, n_skip=0, n_skip_end=0, x_axis='lrs'):
        '''
        Plots the loss function with respect to learning rate, in log scale.
        '''
        plt.ylabel("validation loss")
        plt.xlabel(x_axis)
        plt.plot(self.batch_history[x_axis][n_skip:-(n_skip_end+1)],
                 self.batch_history['loss'][n_skip:-(n_skip_end+1)])
        # plt.xscale('log')
        plt.show()

    def plot_lr(self):
        '''Plots learning rate in jupyter notebook or console, depending on the enviroment of the learner.'''
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(self.batch_history['iteration'], self.batch_history['lrs'])
        plt.title('iteration-learning rate')
        plt.show()


class LR_Updater(PerformanceLogger):
    def __init__(self, base_lr=0.001, verbose=0):
        super(LR_Updater, self).__init__()
        self.base_lr = base_lr
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        old_lr = float(K.get_value(self.model.optimizer.lr))
        self.old_lr = old_lr
        if self.verbose>0:
            print('Reset leraning rate from {0} to {1}'.format(old_lr, self.base_lr))
        K.set_value(self.model.optimizer.lr, self.base_lr)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if self.verbose>0:
            print('Reset leraning rate from {0} to {1}'.format(self.base_lr, self.old_lr))
        K.set_value(self.model.optimizer.lr, self.old_lr)

    def on_batch_end(self, batch, logs=None):
        super().on_batch_end(batch, logs)
        self.update_lr()

    @abstractmethod
    def update_lr(self):
        raise NotImplementedError


class LR_Finder(LR_Updater):
    '''
    Helps you find an optimal learning rate for a model, as per suggetion of 2015 CLR paper.
    Learning rate is increased in linear or log scale, depending on user input, and the result of the loss funciton is retained and can be plotted later.
    '''
    def __init__(self, nb, base_lr=1e-5, max_lr=10, verbose=0):
        assert nb >= 2
        assert max_lr > base_lr
        super(LR_Finder, self).__init__(base_lr, verbose)
        self.max_lr = max_lr
        self.nb = nb
        self.stop_dv = True
        self.lr_interval = (max_lr - base_lr) / (nb - 1)

    def update_lr(self):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        lr = lr + self.lr_interval
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: LR_Finder reducing learning '
                  'rate to %s.' % (self.batch_num + 1, lr))


class CircularLR(LR_Updater):
    '''
     A learning rate updater that implements the CircularLearningRate (CLR) scheme.
    Learning rate is increased then decreased linearly.
    '''
    def __init__(self, step_size=128, base_lr=1e-5, max_lr=10, decay=0.9, decay_type=None, decay_freq=1, verbose=0):
        assert max_lr > base_lr
        if decay_type not in ['exp', None]:
            raise ValueError('Invalid decay type. '
                             'Decay type should be one of '
                             '{"exp", None}')
        super(CircularLR, self).__init__(base_lr, verbose)
        self.max_lr = max_lr
        self.step_size = step_size
        self.decay = decay
        self.decay_type = decay_type
        self.decay_freq = decay_freq
        self.lr_interval = (max_lr - base_lr) / step_size
        self.cycle_num = 0

    def update_lr(self):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if self.decay_type is not None:
            # 已经经历过的周期数
            cycle_num = self.batch_num // (self.step_size * 2)
            # 是否对最大学习率进行（衰减）更新的标志，当经过self.decay_freq周期后，实行一次更新
            decay_flag = self.batch_num % (self.step_size * 2 * self.decay_freq)

            if decay_flag == 0:
                decay_max_lr = self.max_lr * (self.decay ** (cycle_num//self.decay_freq))
                if decay_max_lr < self.base_lr:
                    decay_max_lr = self.base_lr  # 强制max_lr不能小于base_lr
                self.lr_interval = (decay_max_lr - self.base_lr) / self.step_size
        half_cycle_num = self.batch_num // self.step_size
        decrease_phase = half_cycle_num % 2
        lr = float(K.get_value(self.model.optimizer.lr))
        lr = lr + (-1)**decrease_phase * self.lr_interval

        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: CircularLR setting learning '
                  'rate to %s.' % (self.batch_num + 1, lr))


