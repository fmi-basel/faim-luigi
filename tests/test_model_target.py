import pytest
import numpy as np
import scipy

from keras.models import Sequential
from keras.layers import Conv1D
from keras.optimizers import SGD

from faim_luigi.targets.model_targets import KerasModelTarget


def _dummy_model():
    '''
    '''
    model = Sequential()
    model.add(
        Conv1D(1,
               kernel_size=7,
               input_shape=(None, 1),
               padding='same',
               activation='linear'))
    model.compile(loss='mse', optimizer=SGD(lr=0.01))
    return model


def _dummy_data():
    '''
    '''
    inputs = np.random.randn(50, 100, 1)
    targets = scipy.ndimage.gaussian_filter1d(inputs, sigma=1, axis=1)
    return inputs, targets


def test_model_target(tmpdir):
    '''test exists, save and load of the TiffImageTarget.

    '''
    target = KerasModelTarget(tmpdir, 'dummy_model')
    assert not target.exists()

    # train a model and save it
    model = _dummy_model()
    model.summary()
    inputs, targets = _dummy_data()
    model.fit(*_dummy_data(),
              epochs=100,
              callbacks=target.get_modelcheckpoint_callbacks())

    assert target.exists()

    # test loading
    with pytest.raises(FileNotFoundError):
        other_model = target.load(best=True)
    other_model = target.load(best=False)
    other_model.summary()
    assert np.all(other_model.layers[0].get_weights()[0] ==  # compare weights
                  model.layers[0].get_weights()[0])
