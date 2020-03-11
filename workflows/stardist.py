import os
import logging

import luigi
from luigi.util import requires
import numpy as np
from tqdm import tqdm

import stardist
from stardist.models import Config3D, StarDist3D
from stardist.models import Config2D, StarDist2D
from csbdeep.utils import normalize

from faim_luigi.tasks.collectors import DataCollectorTask
from faim_luigi.tasks.mixins import TrainingMixin, AugmentationMixin
from faim_luigi.tasks.utils import task_to_hash

from .stardist_utils import Augmentor

logger = logging.getLogger('luigi-interface')

AXIS_NORM = (0, 1, 2)


def preprocess_image(img):
    '''
    '''
    return normalize(img, .1, 99.9, axis=AXIS_NORM)


class StarDistModelTarget(luigi.LocalTarget):
    '''
    '''
    @property
    def _model_class(self):
        if self.ndim == 2:
            return StarDist2D
        elif self.ndim == 3:
            return StarDist3D
        raise NotImplementedError()

    @property
    def _config_class(self):
        if self.ndim == 2:
            return Config2D
        elif self.ndim == 3:
            return Config3D
        raise NotImplementedError()

    def __init__(self,
                 output_folder,
                 model_name,
                 model_subdir='models',
                 ndim=3,
                 *args,
                 **kwargs):

        if output_folder.split(os.sep)[-1] == model_subdir:
            model_dir = output_folder
        else:
            model_dir = os.path.join(output_folder, model_subdir)

        self.model_name = model_name
        self.model_dir = model_dir
        self.ndim = ndim
        assert self.ndim in [2, 3]
        super().__init__(os.path.join(self.model_dir, self.model_name), *args,
                         **kwargs)

    def construct(self, **config):
        '''
        '''
        return self._model_class(self._config_class(**config),
                                 name=self.model_name,
                                 basedir=self.model_dir)

    def load(self):
        '''
        '''
        self._model_class(None, name=self.model_name, basedir=self.model_dir)


@requires(DataCollectorTask)
class FitStardist3DModel(luigi.Task, TrainingMixin, AugmentationMixin):
    '''
    '''

    # TODO handle the validation data pattern.

    @property
    def _run_hash(self):
        '''
        '''
        return '{}'.format(task_to_hash(self))

    @property
    def model_name(self):
        '''
        '''
        return os.path.join(
            'stardist-L{}-W{}-{}'.format(self.model_levels, self.model_width,
                                         self.head), self._run_hash)

    def output(self):
        '''
        '''
        return StarDistModelTarget(self.output_folder, self.model_name, ndim=3)

    def preprocess_image(self, img):
        '''can be customized if necessary.
        '''
        return preprocess_image(img)

    def _generate_config(self, imgs, annotations):
        '''
        '''
        n_channel = 1 if imgs[0].ndim == 3 else imgs[0].shape[-1]
        extents = stardist.calculate_extents(annotations)
        anisotropy = tuple(np.max(extents) / extents)
        logger.info('Empirical anisotropy of labeled objects = %s', anisotropy)

        # Predict on subsampled grid for increased efficiency and
        # larger field of view

        conf = dict(
            rays=stardist.Rays_GoldenSpiral(self.n_rays,
                                            anisotropy=anisotropy),
            grid=tuple(1 if a > 1.5 else 4 for a in anisotropy),
            anisotropy=anisotropy,
            use_gpu=False,  # GPU for calculating polyhedras -- NOT
            # related to training
            n_channel_in=n_channel,
            train_patch_size=tuple(int(x) for x in self.train_patch_size),
            train_batch_size=self.train_batch_size,
            train_learning_rate=self.train_learning_rate,
            train_steps_per_epoch=max(
                len(imgs) // self.train_batch_size, self.min_steps_per_epoch),
            train_epochs=self.train_epochs,
            resnet_n_blocks=self.n_blocks,
            resnet_n_filter_base=self.n_filter_base,
            train_reduce_lr={
                'patience': 10,
                'factor': 0.5
            })
        return conf

    def _prepare_data(self):
        '''
        '''
        collection = self.input().load()

        def _is_validation(path):
            return self.validation_split_pattern in path

        imgs, annotations, val_mask = zip(
            *[(image.load(), mask.load(), _is_validation(image.path))
              for image in collection.training_images()
              for mask in image.mask])
        if imgs[0].ndim == 4 and annotations[0].ndim == 3:
            annotations = [x[..., None] for x in annotations]

        if len(imgs) <= 0:
            raise RuntimeError('No training data available!')

        # Normalize and clean up annotations
        imgs = [self.preprocess_image(img) for img in tqdm(imgs, ncols=80)]
        annotations = [
            stardist.fill_label_holes(annot)
            for annot in tqdm(annotations, ncols=80)
        ]

        train_img = [
            img for (img, is_val) in zip(imgs, val_mask) if not is_val
        ]
        train_annot = [
            annot for (annot, is_val) in zip(annotations, val_mask)
            if not is_val
        ]

        val_img = [img for (img, is_val) in zip(imgs, val_mask) if is_val]
        val_annot = [
            annot for (annot, is_val) in zip(annotations, val_mask) if is_val
        ]

        return (train_img, train_annot), (val_img, val_annot)

    def run(self):
        '''
        '''
        training_data, validation_data = self._prepare_data()

        print('Training')
        print(training_data[0].shape, training_data[1].shape)
        print('Validation')
        print(validation_data[0].shape, validation_data[1].shape)

        model = self.output().construct(self._generate_config(*training_data))

        augmenter = Augmentor(
            with_flips=self.augmentation_with_flips,
            sigma_mean=self.augmentation_gaussian_noise_mu,
            sigma_std=self.augmentation_gaussian_noise_sigma).run

        model.prepare_for_training()
        model.train(*training_data,
                    validation_data=validation_data,
                    augmenter=augmenter)
        try:
            model.optimize_thresholds(*validation_data)
        except Exception as err:
            print('OOM Error. Trying tiled prediction')

            max_shape = np.max([img.shape for img in validation_data[0]],
                               axis=0)
            n_tiles = tuple(
                int(round(dim / patch_size))
                for dim, patch_size in zip(max_shape, self.train_patch_size))
            model.optimize_thresholds(*validation_data,
                                      predict_kwargs={'n_tiles': n_tiles})
