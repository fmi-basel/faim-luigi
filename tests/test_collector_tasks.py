import pathlib
import fnmatch

import pytest
import luigi
import numpy as np

from faim_luigi.tasks.collectors import FileCollectorTask
from faim_luigi.tasks.collectors import ImageCollectorTask

from faim_luigi.targets.image_target import TiffImageTarget


def _run_task(task):
    '''runs the given luigi task in a new worker and returns
    it's output.

    '''
    worker = luigi.worker.Worker()
    worker.add(task)
    worker.run()
    return task.output()


@pytest.mark.parametrize('pattern', ['*', '*txt'])
def test_file_collector(tmpdir, pattern):
    '''
    '''
    # create some files
    test_dir = tmpdir / 'test_file_collector'
    test_dir.mkdir()
    fnames = ['abc.tif', 'def.jpg', 'ghi.txt', 'ijk.txt']
    included = [fname for fname in fnames if fnmatch.fnmatch(fname, pattern)]

    for fname in fnames:
        path = test_dir / fname
        with open(path, 'w'):
            pass

    # Run the task
    task = FileCollectorTask(input_folder=test_dir, file_pattern=pattern)
    outputs = _run_task(task)

    # check if the collected files match the expected ones.
    collected = [pathlib.Path(target.path) for target in outputs]
    assert all(path.parent == test_dir for path in collected)
    assert all(path.name in included for path in collected)
    assert all(fname in [path.name for path in collected]
               for fname in included)


@pytest.mark.parametrize('pattern', ['*tif'])
def test_image_collector(tmpdir, pattern):
    '''
    '''
    # create some files
    test_dir = tmpdir / 'test_image_collector'
    test_dir.mkdir()
    fnames = ['abc.tif', 'def.jpg', 'ghi.txt', 'ijk.txt']
    included = [fname for fname in fnames if fnmatch.fnmatch(fname, pattern)]

    dummy = np.arange(20).reshape(1, 4, 5)
    for fname in fnames:
        path = test_dir / fname
        target = TiffImageTarget(path).save(dummy)

    # Run the task
    task = ImageCollectorTask(input_folder=test_dir, file_pattern=pattern)
    outputs = _run_task(task)

    # check if the collected files match the expected ones.
    collected = [pathlib.Path(target.path) for target in outputs]
    assert all(path.parent == test_dir for path in collected)
    assert all(path.name in included for path in collected)
    assert all(fname in [path.name for path in collected]
               for fname in included)

    # check presence of image i/o functions
    for target in outputs:
        img = target.load()
        assert np.all(dummy == img)

    assert True
