import numpy as np

AXIS_NORM = (0, 1, 2)


def _random_xy_flip(img, lbl):
    '''
    '''
    assert img.ndim == 3

    for ii in range(0, 3):
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=ii)
            lbl = np.flip(lbl, axis=ii)
    return img, lbl


class Augmentor:
    '''imitates basic image dataset augmentation
    (as in dlutils.dataset.augmentations) for stardist.

    '''
    def __init__(self, with_flips=True, sigma_std=0, sigma_mean=0):
        '''
        '''
        self.with_flips = with_flips
        self.sigma_mean = sigma_mean
        self.sigma_std = sigma_std

    def _add_random_gaussian_noise(self, img):
        '''
        '''
        sigma = np.random.randn() * self.sigma_std + self.sigma_mean
        if sigma <= 0:
            return img
        return img + np.random.randn(*img.shape) * sigma

    def run(self, img_batch, annotation_batch):
        '''Augmentation for data batch.

        '''
        if self.with_flips:
            X_batch, Y_batch = zip(*[
                _random_xy_flip(img, lbl)
                for img, lbl in zip(img_batch, annotation_batch)
            ])

        if self.sigma_std > 0 or self.sigma_mean > 0:
            X_batch = [self._add_random_gaussian_noise(x) for x in X_batch]

        return X_batch, Y_batch
