'''Luigi.Task mixin for data augmentation.

'''
import luigi

from dlutils.dataset.augmentations import random_axis_flip
from dlutils.dataset.augmentations import random_gaussian_noise
from dlutils.dataset.augmentations import random_gaussian_offset


class AugmentationMixin:
    '''Adds parameters for augmentation functions to task.

    '''
    # augmentation params
    augmentation_with_flips = luigi.BoolParameter(default=False)
    augmentation_gaussian_noise_sigma = luigi.FloatParameter(default=0)
    augmentation_gaussian_noise_mu = luigi.FloatParameter(default=0)
    augmentation_offset_sigma = luigi.FloatParameter(default=0)

    # determines which entries are to be treated as inputs
    input_keys = luigi.ListParameter(default=['img'])

    def _get_dimensions(self):
        '''guess along how many dimensions to flip.

        '''
        try:
            return len(self.patch_size)
        except Exception:
            print('Patch size not known. Assuming 2D images.')
            return 2

    def get_augmentations(self):
        '''get a list of augmentation functions parametrized by the given
        values.

        '''
        augmentations = []
        if self.augmentation_with_flips:
            for axis in range(self._get_dimensions()):
                augmentations.append(random_axis_flip(axis, 0.5))

        if self.augmentation_gaussian_noise_mu > 0. or \
           self.augmentation_gaussian_noise_sigma > 0.:
            augmentations.append(
                random_gaussian_noise(self.augmentation_gaussian_noise_mu,
                                      self.augmentation_gaussian_noise_sigma,
                                      self.input_keys))

        if abs(self.augmentation_offset_sigma) >= 1e-8:
            augmentations.append(
                random_gaussian_offset(self.augmentation_offset_sigma,
                                       self.input_keys))

        print('\nAdded augmentations: ')
        for augmentation in augmentations:
            print('\t', augmentation)
        print()
        return augmentations
