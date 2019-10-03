import tensorflow as tf

import luigi


def random_axis_flip(axis, flip_prob):
    '''reverses axis with probability threshold

    '''

    def _flipper(input_dict):
        '''
        '''
        draw_prob = tf.random.uniform(shape=[],
                                      minval=0,
                                      maxval=1,
                                      dtype=tf.float32)
        return {
            key:
            tf.cond(draw_prob >=
                    flip_prob, lambda: tf.reverse(val, [axis]), lambda: val)
            for key, val in input_dict.items()
        }

    return _flipper


def gaussian_noise(noise_mu, noise_sigma, keys):
    '''adds gaussian noise to the given channel.

    Noise levels (mu, sigma) are sampled for each batch from the given
    noise_mu and noise_sigma.

    '''

    def _distorter(input_dict):
        '''
        '''
        sigma = tf.maximum(
            0., tf.random_normal(shape=[], mean=noise_mu, stddev=noise_sigma))

        for key in keys:
            image = input_dict[key]
            noise = tf.random_normal(shape=tf.shape(image),
                                     mean=0,
                                     stddev=sigma)
            input_dict[key] = image + noise
        return input_dict

    return _distorter


def gaussian_offset(offset_sigma, keys):
    '''draws a random offset from N(0, offset_sigma) and
    adds it to the given input[key].
    '''

    def _distorter(input_dict):
        '''
        '''
        for key in keys:
            image = input_dict[key]
            offset = tf.random_normal(shape=[], mean=0, stddev=offset_sigma)
            input_dict[key] = image + offset
        return input_dict

    return _distorter


class AugmentationMixin:
    '''
    '''

    # augmentation params
    augmentation_with_flips = luigi.BoolParameter(default=False)
    augmentation_gaussian_noise_sigma = luigi.FloatParameter(default=0)
    augmentation_gaussian_noise_mu = luigi.FloatParameter(default=0)
    augmentation_offset_sigma = luigi.FloatParameter(default=0)

    keys = ['img']

    def _get_dimensions(self):
        '''
        '''
        try:
            return len(self.patch_size)
        except Exception:
            print('Patch size not known. Assuming 2D images.')
            return 2

    def get_augmentations(self):
        '''
        '''
        augmentations = []
        if self.augmentation_with_flips:
            for axis in range(self._get_dimensions()):
                augmentations.append(random_axis_flip(axis, 0.5))

        if self.augmentation_gaussian_noise_mu > 0. or \
           self.augmentation_gaussian_noise_sigma > 0.:
            augmentations.append(
                gaussian_noise(self.augmentation_gaussian_noise_mu,
                               self.augmentation_gaussian_noise_sigma,
                               self.keys))

        if abs(self.augmentation_offset_sigma) >= 1e-8:
            augmentations.append(
                gaussian_offset(self.augmentation_offset_sigma, self.keys))

        print('\nAdded augmentations: ')
        for augmentation in augmentations:
            print('\t', augmentation)
        print()
        return augmentations
