'''Target for multifile tfrecord dataset.

'''
import os
from glob import glob

from dlutils.dataset.dataset import create_dataset_for_training
from dlutils.dataset.dataset import create_dataset_for_validation


class TFRecordDatasetTarget:
    '''provides a target for a multifile tfrecord dataset.

    TODO Consider adding dataset metadata file that could
    be used to check if the number of files is consistent.

    '''

    _train_subfolder = 'train'
    _val_subfolder = 'val'
    _fname_pattern = '*tfrecord'

    def __init__(self, folder, parser):
        '''
        '''
        self.parser = parser
        self.folder = folder

    def makedirs(self):
        '''
        '''
        for folder in [
                self.folder, self.training_folder, self.validation_folder
        ]:
            os.makedirs(folder, exist_ok=True)

    @property
    def training_folder(self):
        return os.path.join(self.folder, self._train_subfolder)

    @property
    def validation_folder(self):
        return os.path.join(self.folder, self._val_subfolder)

    @property
    def _training_pattern(self):
        return os.path.join(self.training_folder, self._fname_pattern)

    @property
    def _validation_pattern(self):
        return os.path.join(self.validation_folder, self._fname_pattern)

    def load_validation(self, batch_size, **kwargs):
        '''
        '''
        return create_dataset_for_validation(
            self._validation_pattern, batch_size, self.parser.parse, **kwargs)

    def load_training(self, batch_size, patch_size, augmentations=[], **kwargs):
        '''
        '''
        if kwargs.get('transforms') is not None:
            augmentations.extend(kwargs.pop('transforms'))
        return create_dataset_for_training(
            self._training_pattern,
            batch_size,
            self.parser.parse,
            patch_size=patch_size,
            transforms=augmentations,
            **kwargs)

    def load(self, batch_size, patch_size=None, augmentations=[], **kwargs):
        '''returns tf.data.Dataset created from the tfrecords.

        '''
        training_dataset = self.load_training(batch_size, patch_size, augmentations, **kwargs)
        validation_dataset = self.load_validation(batch_size, **kwargs)

        return training_dataset, validation_dataset

    def exists(self):
        '''
        '''
        if (not os.path.exists(self.folder)
                or not os.path.exists(self.training_folder)
                or not os.path.exists(self.validation_folder)):
            return False

        train_records = sum(1 for _ in glob(self._training_pattern))

        if train_records <= 1:
            return False

        return True
