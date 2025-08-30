import numpy as np
import numpy.linalg as la
from odeiter.single_step import RK4
from odeiter.adams_moulton import (
    AM1,
    AM2,
    AM3,
    AM4,
)
from odeiter.time_domain import TimeDomain_Start_Stop_MaxSpacing
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


def exact_solution(t_diff: float, u0: np.ndarray[float]) -> np.ndarray[float]:
    return INIT * np.exp(-t_diff)


def test_AM1_order():
    solver = AM1()
    assert solver.order == 2


def test_AM2_order():
    solver = AM2(seed=RK4(), seed_steps_per_step=2)
    assert solver.order == 3


def test_AM2_name():
    solver = AM2(seed=RK4(), seed_steps_per_step=2)
    assert solver.name == "AM2 (seed: RK4)"


def test_AM2_order_actual():
    solver = AM2(seed=RK4(), seed_steps_per_step=2)
    t_start, t_final = 0, 1
    exact = exact_solution(t_final - t_start, INIT)
    errs = []
    for dt in DTS:
        time = TimeDomain_Start_Stop_MaxSpacing(0, 1, dt)
        approx = solver.t_final(INIT, rhs, time)
        errs.append(la.norm((approx - exact) / exact))
    assert 3.0 == pytest.approx(np.log10(errs[0] / errs[-1]), rel=0.1)


def test_AM3_order():
    solver = AM3(seed=RK4(), seed_steps_per_step=2)
    assert solver.order == 4


def test_AM3_name():
    solver = AM3(seed=RK4(), seed_steps_per_step=2)
    assert solver.name == "AM3 (seed: RK4)"


def test_AM3_order_actual():
    solver = AM3(seed=RK4(), seed_steps_per_step=2)
    t_start, t_final = 0, 1
    exact = exact_solution(t_final - t_start, INIT)
    errs = []
    for dt in DTS:
        time = TimeDomain_Start_Stop_MaxSpacing(0, 1, dt)
        approx = solver.t_final(INIT, rhs, time)
        errs.append(la.norm((approx - exact) / exact))
    assert 4.0 == pytest.approx(np.log10(errs[0] / errs[-1]), rel=0.1)


def test_AM4_order():
    solver = AM4(seed=RK4(), seed_steps_per_step=2)
    assert solver.order == 5


def test_AM4_name():
    solver = AM4(seed=RK4(), seed_steps_per_step=2)
    assert solver.name == "AM4 (seed: RK4)"


def test_AM4_order_actual():
    solver = AM4(seed=RK4(), seed_steps_per_step=2)
    t_start, t_final = 0, 1
    exact = exact_solution(t_final - t_start, INIT)
    errs = []
    for dt in DTS:
        time = TimeDomain_Start_Stop_MaxSpacing(0, 1, dt)
        approx = solver.t_final(INIT, rhs, time)
        errs.append(la.norm((approx - exact) / exact))
    assert 5.0 == pytest.approx(np.log10(errs[0] / errs[-1]), rel=0.1)
