import numpy as np
import numpy.linalg as la
from odeiter.single_step import RK4
from odeiter.adams_bashforth import (
    AB2,
    AB3,
    AB4,
    AB5,
)
from odeiter.time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
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


def test_AB2_order():
    solver = AB2(RK4(), 2)
    assert solver.order == 2


def test_AB2_name():
    solver = AB2(RK4(), 2)
    assert solver.name == "AB2 (seed: RK4)"


def test_AB2_order_actual():
    solver = AB2(RK4(), 2)
    t_start, t_final = 0, 1
    exact = exact_solution(t_final - t_start, INIT)
    errs = []
    for dt in DTS:
        time = TimeDomain_Start_Stop_MaxSpacing(0, 1, dt)
        approx = solver.t_final(INIT, rhs, time)
        errs.append(la.norm((approx - exact) / exact))
    assert 2.0 == pytest.approx(np.log10(errs[0] / errs[-1]), rel=0.1)


def test_AB3_order():
    solver = AB3(RK4(), 2)
    assert solver.order == 3


def test_AB3_name():
    solver = AB3(RK4(), 2)
    assert solver.name == "AB3 (seed: RK4)"


def test_AB3_order_actual():
    solver = AB3(RK4(), 2)
    dts = [0.01, 0.001]
    t_start, t_final = 0, 1
    exact = exact_solution(t_final - t_start, INIT)
    errs = []
    for dt in DTS:
        time = TimeDomain_Start_Stop_MaxSpacing(0, 1, dt)
        approx = solver.t_final(INIT, rhs, time)
        errs.append(la.norm((approx - exact) / exact))
    assert 3.0 == pytest.approx(np.log10(errs[0] / errs[-1]), rel=0.1)


def test_AB4_order():
    solver = AB4(RK4(), 2)
    assert solver.order == 4


def test_AB4_name():
    solver = AB4(RK4(), 2)
    assert solver.name == "AB4 (seed: RK4)"


def test_AB4_order_actual():
    solver = AB4(RK4(), 2)
    dts = [0.01, 0.001]
    t_start, t_final = 0, 1
    exact = exact_solution(t_final - t_start, INIT)
    errs = []
    for dt in DTS:
        time = TimeDomain_Start_Stop_MaxSpacing(0, 1, dt)
        approx = solver.t_final(INIT, rhs, time)
        errs.append(la.norm((approx - exact) / exact))
    assert 4.0 == pytest.approx(np.log10(errs[0] / errs[-1]), rel=0.1)


def test_AB5_order():
    solver = AB5(RK4(), 2)
    assert solver.order == 5


def test_AB5_name():
    solver = AB5(RK4(), 2)
    assert solver.name == "AB5 (seed: RK4)"


def test_AB5_order_actual():
    solver = AB5(RK4(), 2)
    dts = [0.1, 0.01]
    t_start, t_final = 0, 1
    exact = exact_solution(t_final - t_start, INIT)
    errs = []
    for dt in dts:
        time = TimeDomain_Start_Stop_MaxSpacing(0, 1, dt)
        approx = solver.t_final(INIT, rhs, time)
        errs.append(la.norm((approx - exact) / exact))
    assert 5.0 == pytest.approx(np.log10(errs[0] / errs[-1]), rel=0.1)
