import os
import re
from glob import glob

import luigi

from faim_luigi.targets.collection_target import CollectionTarget


class DataCollectorTask(luigi.Task):
    '''
    '''
    input_folder = luigi.Parameter()
    output_folder = luigi.Parameter()

    image_pattern = luigi.Parameter()
    annotation_sub_pattern = luigi.ListParameter()
    test_split_pattern = luigi.Parameter()

    def image_locator(self):
        '''
        '''
        for path in sorted(
                glob(os.path.join(self.input_folder, self.image_pattern))):
            yield os.path.split(path)

    def annotation_locator(self, image_folder, image_fname):
        '''
        '''
        candidate_path = re.sub(*self.annotation_sub_pattern,
                                os.path.join(image_folder, image_fname))
        if os.path.exists(candidate_path):
            yield os.path.split(candidate_path)

    def is_in_training_split(self, folder, fname):
        '''Returns True if the test_split_pattern does NOT match the
        given folder/fname

        '''
        match = re.search(self.test_split_pattern, os.path.join(folder, fname))
        if match:
            return False
        return True

    def run(self):
        '''
        '''
        try:
            self.output().makedirs()
            collection = self.output().construct(self.image_locator,
                                                 self.annotation_locator,
                                                 self.is_in_training_split)

            total_images = sum(1 for image in collection)
            annotated_images = sum(1 for image in collection if image.mask)

            print('Found {} images, {} with annotations'.format(
                total_images, annotated_images))

            if total_images <= 0:
                raise RuntimeError('No image data found!')

        except Exception:
            self.output().remove()
            raise

    def output(self):
        '''
        '''
        return CollectionTarget(os.path.join(self.output_folder, 'dataset.db'))
