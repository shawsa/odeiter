import numpy as np
import numpy.linalg as la
from odeiter.time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
from odeiter.time_integrator import TqdmWrapper
from odeiter.single_step import (
    Euler,
    EulerDelta,
    RK4,
    default_root_finder,
    ImplicitEuler,
    Trapezoidal,
)
import pytest


INIT = np.array([1.0, 2.0])


def rhs(t: float, u: np.ndarray[float]) -> np.ndarray[float]:
    """The right-hand-side of the equation
    u' = u

    which has the solution
    u(t) = u(0)*e^t
    """
    return u


def test_solution_generator():
    solver = Euler()
    time = TimeDomain(0.0, 0.1, 10)
    solution = solver.solution_generator(u0=INIT, rhs=rhs, time=time)
    _ = next(solution)
    first_step = next(solution)
    assert np.all(first_step == np.array([1.1, 2.2]))


def test_t_Final():
    solver = Euler()
    time = TimeDomain(0.0, 0.1, 1)
    final = solver.t_final(u0=INIT, rhs=rhs, time=time)
    assert np.all(final == np.array([1.1, 2.2]))


def test_tqdm_wrapper():
    solver = TqdmWrapper(Euler())
    time = TimeDomain(0.0, 0.1, 10)
    for sol1, sol2 in zip(
        TqdmWrapper(Euler()).solve(INIT, rhs, time),
        Euler().solve(INIT, rhs, time),
    ):
        assert np.all(sol1 == sol2)
    assert np.all(
        TqdmWrapper(Euler()).t_final(INIT, rhs, time)
        == Euler().t_final(INIT, rhs, time)
    )

    solution = solver.solution_generator(u0=INIT, rhs=rhs, time=time)
    _ = next(solution)
    first_step = next(solution)
    assert np.all(first_step == np.array([1.1, 2.2]))


def test_Euler_order():
    solver = Euler()
    assert solver.order == 1


def test_Euler_name():
    solver = Euler()
    assert solver.name == "Euler"


def test_Euler_update():
    solver = Euler()
    time = TimeDomain(0.0, 0.1, 10)
    first_step = solver.update(t=0.0, u=INIT, f=rhs, h=time.spacing)
    assert np.all(first_step == np.array([1.1, 2.2]))


DELTA_TIME = 0.3
DELTA_PROFILE = np.array([1.0, -1.0])


def test_EulerDelta_init():
    solver = EulerDelta(DELTA_TIME, DELTA_PROFILE)
    assert solver.delta_time == 0.3
    assert np.all(solver.delta_profile == DELTA_PROFILE)


def test_EulerDelta_order():
    solver = EulerDelta(DELTA_TIME, DELTA_PROFILE)
    assert solver.order == 1


def test_EulerDelta_name():
    solver = EulerDelta(DELTA_TIME, DELTA_PROFILE)
    assert solver.name == "Euler-Delta"


def test_EulerDelta_update():
    euler = Euler()
    delta = EulerDelta(DELTA_TIME, DELTA_PROFILE)
    time = TimeDomain(0.0, 0.1, 10)
    euler_solution = euler.solve(INIT, rhs, time)
    delta_solution = delta.solve(INIT, rhs, time)
    delta_index = 3
    assert abs(time.array[delta_index] - delta.delta_time) < time.spacing / 2
    for index in range(delta_index):
        assert np.all(euler_solution[index] == delta_solution[index])
    assert np.all(
        euler_solution[delta_index + 1] + delta.delta_profile
        == delta_solution[delta_index + 1]
    )


def test_RK4_order():
    solver = RK4()
    assert solver.order == 4


def test_RK4_name():
    solver = RK4()
    assert solver.name == "RK4"


def test_RK4_order_actual():
    solver = RK4()
    dts = [0.01, 0.001]
    t_start, t_final = 0, 1
    exact = INIT * np.exp(t_final - t_start)
    errs = []
    for dt in dts:
        time = TimeDomain_Start_Stop_MaxSpacing(0, 1, dt)
        approx = solver.t_final(INIT, rhs, time)
        errs.append(la.norm((approx - exact) / exact))
    assert 4 == pytest.approx(np.log10(errs[0] / errs[-1]), rel=0.1)


def test_default_root_finder():
    def foo(x):
        return x**2 - 1

    assert 1.0 == pytest.approx(default_root_finder(foo, 1.1))


def test_default_root_finder_failure():
    def foo(x):
        return x**2 + 1

    with pytest.raises(ValueError):
        default_root_finder(foo, 1.0)


def test_implicit_euler_order():
    solver = ImplicitEuler()
    assert solver.order == 1


def test_implicit_euler_name():
    solver = ImplicitEuler()
    assert solver.name == "Implicit Euler"


def test_implicit_euler_order_actual():
    solver = ImplicitEuler()
    dts = [0.01, 0.001]
    t_start, t_final = 0, 1
    exact = INIT * np.exp(t_final - t_start)
    errs = []
    for dt in dts:
        time = TimeDomain_Start_Stop_MaxSpacing(0, 1, dt)
        approx = solver.t_final(INIT, rhs, time)
        errs.append(la.norm((approx - exact) / exact))
    assert 1.0 == pytest.approx(np.log10(errs[0] / errs[-1]), rel=0.1)


def test_trapezoidal_order():
    solver = Trapezoidal()
    assert solver.order == 2


def test_trapezoidal_name():
    solver = Trapezoidal()
    assert solver.name == "Trapezoidal"


def test_trapezoidal_order_actual():
    solver = Trapezoidal()
    dts = [0.01, 0.001]
    t_start, t_final = 0, 1
    exact = INIT * np.exp(t_final - t_start)
    errs = []
    for dt in dts:
        time = TimeDomain_Start_Stop_MaxSpacing(0, 1, dt)
        approx = solver.t_final(INIT, rhs, time)
        errs.append(la.norm((approx - exact) / exact))
    assert 2.0 == pytest.approx(np.log10(errs[0] / errs[-1]), rel=0.1)
