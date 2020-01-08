import os
import abc

import luigi

from dlutils.models import load_model
from dlutils.training.callbacks import ModelConfigSaver, ModelCheckpoint


class TrainableModelTargetMixin(abc.ABC):
    '''
    '''

    @abc.abstractmethod
    def get_modelcheckpoint_callbacks(self):
        '''returns a list of callbacks for training that
        will save the model.

        '''
        pass

    @abc.abstractmethod
    def load(self):
        '''loads a trained model.

        '''
        pass


class KerasModelTarget(luigi.LocalTarget):
    '''

    TODO this class does not yet handle the atomic write.

    '''

    weights_fname = {'best': 'model_best.h5', 'latest': 'model_latest.h5'}
    architecture_fname = 'model_architecture.yaml'

    def __init__(self,
                 output_folder,
                 model_name,
                 model_subdir='models',
                 *args,
                 **kwargs):
        '''
        '''
        path = os.path.join(output_folder, model_subdir, model_name)
        self.model_name = model_name
        super().__init__(path, *args, **kwargs)

    def exists(self):
        '''returns true if the architecture.yaml and ONE of the weight
        files exists.

        Note that this method cannot check whether everything in the
        training went correctly.

        '''
        if not super().exists():
            return False
        return any(
            os.path.exists(os.path.join(self.path, fname))
            for weights in self.weights_fname.values()
            for fname in [weights, self.architecture_fname])

    def get_modelcheckpoint_callbacks(self):
        '''returns a list of callbacks for training that
        will save the model.

        '''
        return [
            ModelConfigSaver(os.path.join(self.path, self.architecture_fname)),
        ] + [
            ModelCheckpoint(
                os.path.join(self.path, fname),
                save_best_only=key == 'best',
                save_weights_only=True)
            for key, fname in self.weights_fname.items()
        ]

    def load(self, best=False):
        '''
        '''
        path = os.path.join(self.path,
                            self.weights_fname['best' if best else 'latest'])
        if not os.path.exists(path):
            raise FileNotFoundError(
                'The weights file for best={}: {} does not exist!'.format(
                    best, self.weights_fname['best' if best else 'latest']))
        return load_model(path)
