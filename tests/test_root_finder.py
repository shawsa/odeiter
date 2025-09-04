from odeiter.root_finder import DefaultRootFinder
import numpy as np
import pytest


def test_default_root_finder_value_error():
    def func(u0):
        return 1 + u0**2

    with pytest.raises(ValueError):
        DefaultRootFinder.solve(func, np.array([1.0, 1.0]))
