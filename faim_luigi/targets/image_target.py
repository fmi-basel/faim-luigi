import abc
import luigi

from skimage.external.tifffile import imread, imsave


class ImageTargetBase(abc.ABC):
    @abc.abstractmethod
    def load(self):
        '''
        '''
        pass

    @abc.abstractmethod
    def save(self, vals):
        '''
        '''
        pass


class TiffImageTarget(luigi.LocalTarget, ImageTargetBase):
    '''provides load and save utilities for local tiff image file targets.

    '''

    def load(self):
        '''
        '''
        return imread(self.path)

    def save(self, vals):
        '''
        '''
        with self.temporary_path() as path:
            imsave(path, vals)
