from glob import glob
import os
import logging

import luigi

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean

DeclarativeBase = declarative_base()


class DataItemMixin:
    '''
    '''
    folder = Column(String, nullable=False)
    fname = Column(String, nullable=False)

    @hybrid_property
    def path(self):
        return self.folder + os.sep + self.fname

    def load(self):
        '''
        '''
        from skimage.external.tifffile import imread
        import numpy as np
        try:
            img = np.concatenate(
                [imread(path)[None] for path in sorted(glob(self.path))],
                axis=0)
            if img.ndim >= 4 and img.shape[0] == 1:
                img = img.squeeze(axis=0)
            return img
        except Exception as err:
            logging.getLogger(__name__).error(
                'Failed to load image from %s: %s', self.path, err)
            raise


class ImageData(DeclarativeBase, DataItemMixin):
    '''
    '''
    __tablename__ = 'ImageData'
    id = Column(Integer, primary_key=True)
    mask = relationship('ImageAnnotationData', back_populates='image')
    is_in_training = Column(Boolean)

    def __repr__(self):
        '''
        '''
        name = 'ID {}: {}'.format(self.id, self.path)
        for mask in self.mask:
            name += '\n  {}'.format(mask)
        return name


class ImageAnnotationData(DeclarativeBase, DataItemMixin):
    '''
    '''
    __tablename__ = 'ImageAnnotationData'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey(ImageData.id))
    image = relationship('ImageData', back_populates='mask')

    def __repr__(self):
        '''
        '''
        return 'Annotation: {}'.format(self.path)


class Collection:
    '''handles an image data collection with annotations.

    '''

    def __init__(self, path):
        '''

        Parameters
        ----------
        output : string
            path to sqlite database.

        '''
        self._logger = logging.getLogger(__name__)
        self._engine = sqlalchemy.create_engine('sqlite:///{}'.format(path))
        self._sessionmaker = sessionmaker(bind=self._engine)

    def _get_session(self):
        '''
        '''
        return self._sessionmaker()

    def build(self, image_locator_fn, mask_locator_fn, split_fn):
        '''
        '''
        self._logger.debug('Creating new database')

        DeclarativeBase.metadata.create_all(self._engine)
        session = self._get_session()

        # collect images and store them in database
        for folder, fname in image_locator_fn():
            image = ImageData(
                folder=folder,
                fname=fname,
                is_in_training=split_fn(folder, fname))
            image.mask = [
                ImageAnnotationData(folder=folder, fname=fname)
                for folder, fname in mask_locator_fn(image.folder, image.fname)
            ]
            session.add(image)

        session.commit()

    def __iter__(self):
        '''
        '''
        session = self._get_session()
        for image in session.query(ImageData).order_by(ImageData.id):
            yield image

    def annotated_images(self):
        '''returns paths to image and annotation.

        '''
        session = self._get_session()
        for image in session.query(ImageData).join(
                ImageAnnotationData).order_by(ImageData.id):
            yield image

    def training_images(self):
        '''
        '''
        session = self._get_session()
        for image in session.query(ImageData).join(
                ImageAnnotationData).order_by(ImageData.id):
            if image.is_in_training:
                yield image


class CollectionTarget(luigi.LocalTarget):
    '''provides an abstraction for loading local Collection targets.

    '''

    def construct(self, *args, **kwargs):
        '''all arguments are passed to Collection.build

        '''
        collection = Collection(self.path)
        collection.build(*args, **kwargs)
        return collection

    def load(self):
        '''
        '''
        if not self.exists():
            raise FileNotFoundError()
        return Collection(self.path)
