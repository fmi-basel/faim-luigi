'''
'''
import os

import luigi
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TerminateOnNaN
from keras.callbacks import EarlyStopping

from dlutils.training.scheduler import CosineAnnealingSchedule


def common_callbacks(output_folder,
                     lr_min,
                     lr_max,
                     epochs,
                     n_restarts=1,
                     restart_decay=0.5,
                     patience=None):
    '''creates several keras callbacks to be used in model.fit or
    model.fit_generator.

    '''
    n_restarts = max(1, n_restarts)
    epochs_to_restart = epochs / n_restarts

    # TODO Add restarts as parameter

    callbacks = [
        LearningRateScheduler(
            CosineAnnealingSchedule(lr_max=lr_max,
                                    lr_min=lr_min,
                                    epoch_max=epochs_to_restart,
                                    reset_decay=restart_decay)),
        TerminateOnNaN(),
        TensorBoard(os.path.join(output_folder, 'tensorboard-logs'),
                    write_graph=True,
                    write_grads=False,
                    write_images=False,
                    histogram_freq=0)
    ]

    if patience >= 1 and patience is not None:
        callbacks.append(EarlyStopping(patience=patience))

    return callbacks


class TrainingMixin:
    '''Adds for default training parameters as luigi task parameters
    and provides the common_callbacks() method.

    '''
    train_learning_rate = luigi.FloatParameter()
    train_learning_rate_min = luigi.FloatParameter()
    train_patience = luigi.IntParameter()
    train_epochs = luigi.IntParameter()

    def common_callbacks(self, output_folder):
        '''creates several keras callbacks to be used in model.fit or
        model.fit_generator.

        '''
        return common_callbacks(output_folder=output_folder,
                                lr_min=self.train_learning_rate_min,
                                lr_max=self.train_learning_rate,
                                patience=self.train_patience,
                                epochs=self.train_epochs)
