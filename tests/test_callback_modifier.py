import numpy as np
from odeiter import Euler, TimeDomain
from odeiter.callback_modifier import callback_modifier
import pytest

INIT = np.array([1.0, 2.0])
DTS = [0.1, 0.01]


def rhs(t: float, u: np.ndarray[float]) -> np.ndarray[float]:
    """The right-hand-side of the equation
    u' = -u

    which has the solution
    u(t) = u(0)*e^-t
    """
    return -u


class CallbackTest:
    def __init__(self):
        self.was_called = False

    def __call__(self, t, u, f, h):
        self.was_called = True
        return t, u, f, h


def test_callback():
    callback_test = CallbackTest()
    solver = callback_modifier(callback_test)(Euler)()
    time = TimeDomain(0, 1, 1)
    solution = solver.t_final(INIT, rhs, time)
    euler_solution = Euler().t_final(INIT, rhs, time)
    assert callback_test.was_called
    assert 0.0 == pytest.approx(solution - euler_solution)
