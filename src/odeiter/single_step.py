"""A collection of single-step solvers."""
from .root_finder import RootFinder, DefaultRootFinder
from .time_domain import TimeDomain
from .time_integrator import TimeIntegrator
from abc import abstractmethod
import numpy as np
from typing import Callable, Generator


class SingleStepMethod(TimeIntegrator):
    """A base class for single step solvers. All single step solvers
    inherit these methods as well as the methods from
    [`time_integrator.TimeIntegrator`](time_integrator.md).
    """

    @abstractmethod
    def update(
        self,
        t: float,
        u: np.ndarray[float],
        f: np.ndarray[float],
        delta_t: float,
    ) -> np.ndarray[float]:
        """
        Compute the next time step. You probably want `solution_generator` instead.

        Parameters:
            t: The current time.
            u: The solution at the current time-step.
            f: the right-hand-side at at the current time-step.
            delta_t: the temporal step-size.

        Returns:
            The solution at the next time step.
        """
        ...

    def solution_generator(
        self,
        u0: np.ndarray[float],
        rhs: Callable[[float, np.ndarray[float]], np.ndarray[float]],
        time: TimeDomain,
    ) -> Generator[np.ndarray[float], None, None]:
        """Create a generator that yields the solution for each time in `time`.

        Parameters:
            u0: The initial condition of the system.
                Must be the same size as the system.
            rhs: The right-hand-side as a function with signature `rhs(t, u) -> u'`.
            time: The discretized time domain from.

        Returns:
            A generator that yields the solution at each time in `time.array`.
        """
        u = u0
        yield u
        for t in time.array[:-1]:
            u = self.update(t, u, rhs, time.spacing)
            yield u


class Euler(SingleStepMethod):
    "Forward Euler, a first order explicit method."

    @property
    def order(self):
        return 1

    @property
    def name(self):
        return "Euler"

    def update(self, t, u, f, delta_t):
        return u + delta_t * f(t, u)


class EulerDelta(SingleStepMethod):
    """Forward Euler, but analytically integrates a delta
    forcing term at a single point in time.
    """

    def __init__(self, delta_time: float, delta_profile: np.ndarray[float]):
        """
        Parameters:
            delta_time: the time at which the delta function is non-zero.
            delta_profile: the magnitude of the delta for each
                dimesion of the system.
        """
        self.delta_time = delta_time
        self.delta_profile = delta_profile

    @property
    def order(self):
        return 1

    @property
    def name(self):
        return "Euler-Delta"

    def update(self, t, u, f, delta_t):
        u_new = u + delta_t * f(t, u)
        if abs(t - self.delta_time) < delta_t / 2:
            u_new += self.delta_profile
        return u_new


class RK4(SingleStepMethod):
    """
    Runge-Kutta order 4.

    A single-step method that performs 4 function evaluations per time
    step. Often used as a default because of it's robust domain of stability
    and it's high order of accuracy.
    """

    @property
    def name(self):
        return "RK4"

    @property
    def order(self):
        return 4

    def update(self, t, u, f, delta_t):
        k1 = f(t, u)
        k2 = f(t + delta_t / 2, u + delta_t / 2 * k1)
        k3 = f(t + delta_t / 2, u + delta_t / 2 * k2)
        k4 = f(t + delta_t, u + delta_t * k3)
        return u + delta_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# implicit methods


class ImplicitEuler(SingleStepMethod):
    """An implicit method of order 1."""

    def __init__(self, root_finder: RootFinder = DefaultRootFinder):
        """
        Parameters:
            root_finder: A RootFinder object to solve the system
                at each time step.
        """
        self.root_finder = root_finder.solve

    @property
    def order(self):
        return 1

    @property
    def name(self):
        return "Implicit Euler"

    def update(self, t, u, f, delta_t):
        def func(x):
            return -x + u + delta_t * f(t + delta_t, x)

        return self.root_finder(func, u)


class Trapezoidal(SingleStepMethod):
    """An implicit method of order 2."""

    def __init__(self, root_finder: RootFinder = DefaultRootFinder):
        """
        Parameters:
            root_finder: A RootFinder object to solve the system
                at each time step.
        """
        self.root_finder = root_finder.solve

    @property
    def order(self):
        return 2

    @property
    def name(self):
        return "Trapezoidal"

    def update(self, t, u, f, delta_t):
        fn = f(t, u)

        def func(x):
            return -x + u + delta_t / 2 * (fn + f(t + delta_t, x))

        return self.root_finder(func, u)
