import abc
import logging
import concurrent.futures

import luigi

from dlutils.dataset.tfrecords import tfrecord_from_iterable

from faim_luigi.targets.tfrecord_dataset import TFRecordDatasetTarget


def _single_tfrecord_prep(sample, target, parser):
    '''

    NOTE Make sure the sample is only loaded at
    time of access through __iter__. Otherwise, the IO will happen
    outside parallelization.

    NOTE target is used to handle atomic write.

    '''
    if target.exists():
        return
    with target.temporary_path() as path:
        tfrecord_from_iterable(path, sample, parser.serialize)


class BaseTFRecordDatasetPreparationTask(luigi.Task, abc.ABC):
    '''TODO docstring
    '''

    # Parameters
    output_folder = luigi.Parameter()
    n_processes = luigi.IntParameter(default=4)

    # Other class attributes
    logger = logging.getLogger('luigi-interface')

    @property
    @abc.abstractmethod
    def parser(self):
        '''return a RecordParser object.
        '''
        pass

    @abc.abstractmethod
    def get_sample_iter(self):
        '''should return sample and a target.

        E.g. yield (img_handle, mask_handle), LocalTarget(target_path)
        '''
        pass

    def _submit(self, pool, sample, target):
        '''allows customization of the work function (which needs to be pickled
        for parallelization.

        '''
        return pool.submit(
            _single_tfrecord_prep,
            sample,
            target,
            parser=self.parser,
        )

    def run(self):
        '''
        '''
        failed = 0

        # create dataset folder structure.
        self.output().makedirs()

        # NOTE we're using concurrent.futures here instead of spawning
        # sub-luigi.Tasks because this makes it much easier to pass
        # non-basic parameters (e.g. the RecordParser) between them.
        with concurrent.futures.ProcessPoolExecutor(self.n_processes) as pool:

            futures = [
                self._submit(pool, sample, target)
                for sample, target in self.get_sample_iter()
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as err:
                    self.logger.error(str(err))
                    failed += 1

        if failed >= 1:
            raise RuntimeError(
                '{} encountered {} error{}. See log for details.'.format(
                    self.__class__.__name__, failed, 's'
                    if failed >= 2 else ''))

    def output(self):
        '''
        '''
        return TFRecordDatasetTarget(self.output_folder, self.parser)
