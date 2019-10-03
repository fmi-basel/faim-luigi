import os
import luigi

from dlutils.models import load_model


class KerasModelTarget(luigi.LocalTarget):
    '''
    '''

    _model_subdir = 'models'
    weights_fname = {'best': 'model_best.h5', 'latest': 'model_latest.h5'}
    architecture_fname = 'model_architecture.yaml'

    def __init__(self, output_folder, model_name, *args, **kwargs):
        '''
        '''
        path = os.path.join(output_folder, self._model_subdir, model_name)
        self.model_name = model_name
        super().__init__(path, *args, **kwargs)

    def exists(self):
        '''
        '''
        if not super().exists():
            return False
        return all(
            os.path.exists(os.path.join(self.path, fname))
            for weights in self.weights_fname.values()
            for fname in [weights, self.architecture_fname])

    def load(self, best=True):
        '''
        '''
        return load_model(
            os.path.join(self.path,
                         self.weights_fname['best' if best else 'latest']))
