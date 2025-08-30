import numpy as np
import numpy.linalg as la
from odeiter.single_step import RK4
from odeiter.backward_differentiation import (
    BDF1,
    BDF2,
    BDF3,
    BDF4,
    BDF5,
    BDF6,
)
from odeiter.time_domain import TimeDomain_Start_Stop_MaxSpacing
import pytest

INIT = np.array([1.0, 2.0])
DTS = [0.1, 0.05]


def rhs(t: float, u: np.ndarray[float]) -> np.ndarray[float]:
    """The right-hand-side of the equation
    u' = -u

    which has the solution
    u(t) = u(0)*e^-t
    """
    return -u


def exact_solution(t_diff: float, u0: np.ndarray[float]) -> np.ndarray[float]:
    return INIT * np.exp(-t_diff)


def test_BDF1():
    solver = BDF1()
    assert solver.order == 1


SEED = RK4()
SEED_STEPS = 2
SOLVERS = [
    solver_class(seed=SEED, seed_steps_per_step=SEED_STEPS)
    for solver_class in [
        BDF2,
        BDF3,
        BDF4,
        BDF5,
        BDF6,
    ]
]


def test_BDF_orders():
    for solver, order in zip(SOLVERS, range(2, 7)):
        assert solver.order == order


def test_BDF_names():
    for solver, order in zip(SOLVERS, range(2, 7)):
        assert solver.name == f"BDF{order} (seed: RK4)"


def test_BDF_order_actual():
    for solver, order in zip(SOLVERS, range(2, 7)):
        t_start, t_final = 0, 1
        exact = exact_solution(t_final - t_start, INIT)
        errs = []
        for dt in DTS:
            time = TimeDomain_Start_Stop_MaxSpacing(0, 1, dt)
            approx = solver.t_final(INIT, rhs, time)
            errs.append(la.norm((approx - exact) / exact))
        assert order == pytest.approx(
            np.log(errs[0] / errs[-1]) / np.log(DTS[0] / DTS[1]), rel=0.1
        )
