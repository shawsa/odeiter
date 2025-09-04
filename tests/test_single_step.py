import numpy as np
import numpy.linalg as la
from odeiter.time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
from odeiter.time_integrator import TqdmWrapper
from odeiter.single_step import (
    Euler,
    EulerDelta,
    RK4,
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


class TestSingleStep:
    def test_solution_generator(self):
        solver = Euler()
        time = TimeDomain(0.0, 0.1, 10)
        solution = solver.solution_generator(u0=INIT, rhs=rhs, time=time)
        _ = next(solution)
        first_step = next(solution)
        assert np.all(first_step == np.array([1.1, 2.2]))

    def test_t_Final(self):
        solver = Euler()
        time = TimeDomain(0.0, 0.1, 1)
        final = solver.t_final(u0=INIT, rhs=rhs, time=time)
        assert np.all(final == np.array([1.1, 2.2]))

    def test_tqdm_wrapper(self):
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


class TestEuler:
    def test_Euler_order(self):
        solver = Euler()
        assert solver.order == 1

    def test_Euler_name(self):
        solver = Euler()
        assert solver.name == "Euler"

    def test_Euler_update(self):
        solver = Euler()
        time = TimeDomain(0.0, 0.1, 10)
        first_step = solver.update(t=0.0, u=INIT, f=rhs, delta_t=time.spacing)
        assert np.all(first_step == np.array([1.1, 2.2]))


DELTA_TIME = 0.3
DELTA_PROFILE = np.array([1.0, -1.0])


class TestEulerDelta:
    def test_EulerDelta_init(self):
        solver = EulerDelta(DELTA_TIME, DELTA_PROFILE)
        assert solver.delta_time == 0.3
        assert np.all(solver.delta_profile == DELTA_PROFILE)

    def test_EulerDelta_order(self):
        solver = EulerDelta(DELTA_TIME, DELTA_PROFILE)
        assert solver.order == 1

    def test_EulerDelta_name(self):
        solver = EulerDelta(DELTA_TIME, DELTA_PROFILE)
        assert solver.name == "Euler-Delta"

    def test_EulerDelta_update(self):
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


class TestRK4:
    def test_RK4_order(self):
        solver = RK4()
        assert solver.order == 4

    def test_RK4_name(self):
        solver = RK4()
        assert solver.name == "RK4"

    def test_RK4_order_actual(self):
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


class TestImplicitEuler:
    def test_implicit_euler_order(self):
        solver = ImplicitEuler()
        assert solver.order == 1

    def test_implicit_euler_name(self):
        solver = ImplicitEuler()
        assert solver.name == "Implicit Euler"

    def test_implicit_euler_order_actual(self):
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


class TestTrapezoidal:
    def test_trapezoidal_order(self):
        solver = Trapezoidal()
        assert solver.order == 2

    def test_trapezoidal_name(self):
        solver = Trapezoidal()
        assert solver.name == "Trapezoidal"

    def test_trapezoidal_order_actual(self):
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
