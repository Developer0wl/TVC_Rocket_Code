import gymnasium
import numpy

import tvc


def test_imports():
    assert tvc.__version__
    assert numpy.__version__
    assert gymnasium.__version__
