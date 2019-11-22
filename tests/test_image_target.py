import pytest
import numpy as np

from faim_luigi.targets.image_target import TiffImageTarget


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_image_target(tmpdir, ndim):
    '''test exists, save and load of the TiffImageTarget.

    '''
    base = 3
    img_shape = tuple(range(base, base + ndim))
    img = np.arange(np.prod(img_shape)).reshape(img_shape)

    target = TiffImageTarget(tmpdir /
                             'tiff_image_target_ndim_{}.tif'.format(ndim))
    assert not target.exists()

    target.save(img)
    assert target.exists()

    loaded = target.load()
    assert np.all(img == loaded)
