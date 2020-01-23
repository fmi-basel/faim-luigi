import os

import numpy as np
import tensorflow as tf

from luigi.util import requires
import luigi

from dlutils.dataset.tfrecords import ImageToSegmentationRecordParser

from faim_luigi.tasks.collectors import DataCollectorTask
from .base import BaseTFRecordDatasetPreparationTask


class LazySampleLoader:
    '''defers loading of image and mask to processing time.

    '''

    def __init__(self, image_handle, mask_handle):
        self.image_handle = image_handle
        self.mask_handle = mask_handle

    def __iter__(self):
        img = self.image_handle.load().astype('uint16')
        mask = self.mask_handle.load().astype('uint8')
        yield (img, mask)


@requires(DataCollectorTask)
class SegmentationTFRDatasetPreparationTask(
        BaseTFRecordDatasetPreparationTask):
    '''
    '''

    validation_ratio = luigi.FloatParameter(default=0.15)

    parser = ImageToSegmentationRecordParser(tf.uint16, tf.uint8, 3)

    def get_sample_iter(self):
        '''
        '''
        collection = self.input().load()

        image_ids = [image.id for image in collection.annotated_images()]
        if not image_ids:
            raise RuntimeError('No annotated images found!')

        validation_ids = np.random.choice(
            image_ids,
            replace=False,
            size=max(1, int(self.validation_ratio * len(image_ids))))

        self.logger.info(
            'Splitting {} samples into {} training and {} validation '.format(
                len(image_ids),
                len(image_ids) - len(validation_ids), len(validation_ids)))

        for img_handle, mask_handle in ((
                image, mask) for image in collection.annotated_images()
                                        for mask in image.mask):

            target = luigi.LocalTarget(
                os.path.join(
                    self.output().validation_folder
                    if img_handle.id in validation_ids else
                    self.output().training_folder, '{:05}.tfrecord'.format(
                        img_handle.id)))

            yield LazySampleLoader(img_handle, mask_handle), target
