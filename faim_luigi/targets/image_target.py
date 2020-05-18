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
    def save(self, vals, **kwargs):
        '''
        '''
        pass


class TiffImageTarget(luigi.LocalTarget, ImageTargetBase):
    '''provides load and save utilities for local tiff image file targets.

    Uses skimage.external.tifffile.

    '''

    def load(self):
        '''
        '''
        return imread(self.path)

    def save(self, vals, **kwargs):
        '''
        '''
        with self.temporary_path() as path:
            imsave(path, vals, **kwargs)
