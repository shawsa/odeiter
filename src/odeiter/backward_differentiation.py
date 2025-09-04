from .root_finder import RootFinder, DefaultRootFinder
from .single_step import ImplicitEuler
from .time_domain import TimeDomain, TimeRay
from .time_integrator import TimeIntegrator

from abc import abstractproperty
from collections import deque
from itertools import islice
import numpy as np
from typing import Callable, Generator


class BackwardDifferentiationAbstract(TimeIntegrator):
    """
    An abstract class for Backward Differentiation (BDF) solvers.
    All BDF solvers will inherit these methods. Additionally,
    they inherit the methods of [`time_integrator.TimeIntegrator`](time_integrator.md)
    as well.
    """

    def __init__(
        self,
        seed: TimeIntegrator,
        seed_steps_per_step: int,
        root_finder: RootFinder = DefaultRootFinder,
    ):
        """
        Parameters:
            seed: another time integrator used to take the first few steps.
            seed_steps_per_step: the number of seed steps taking per step
                of the AB integrator.
            root_finder: a function of the form `root_finder(f, u0)` that returns
                the solution to `f(u) = 0` using `u0` as an approximate soluiton.
        """
        self.seed = seed
        self.seed_steps_per_step = seed_steps_per_step
        self.seed_steps = self.order - 1
        self.root_finder = root_finder.solve

    @abstractproperty
    def name(self) -> str:
        """
        Returns:
            The name of the method
        """
        ...

    @abstractproperty
    def order(self) -> int:
        """
        Returns:
            The order of the method
        """
        ...

    @abstractproperty
    def us_coeffs(self) -> list[float]:
        """
        Returns:
            The solution interpolation coefficients of the method.
        """
        ...

    @abstractproperty
    def f_coeff(self) -> float:
        """
        Returns:
            The derivative interpolation coefficient of the method.
        """
        ...

    def update(
        self,
        t: float,
        us: np.ndarray[float],
        rhs: Callable[[float, np.ndarray[float]], np.ndarray[float]],
        delta_t: float,
    ) -> np.ndarray[float]:
        """
        Compute the next time step. You probably want `solution_generator` instead.

        Parameters:
            t: The current time.
            us: The solution at the current time-step and several previous time-steps.
            rhs: The right-hand-side of the system as a function `rhs(t, u) -> u'`.
            delta_t: the temporal step-size.

        Returns:
            The solution at the next time step.
        """
        const_vec = sum(c * u for c, u in zip(self.us_coeffs[::-1], us))

        def func(x):
            return -x + const_vec + delta_t * self.f_coeff * rhs(t + delta_t, x)

        return self.root_finder(func, us[-1])

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
        seed_steps = self.order - 1
        seed_time = TimeRay(
            time.start,
            time.spacing / self.seed_steps_per_step,
        )
        t = time.start
        us = deque(maxlen=self.order)
        for u in islice(
            self.seed.solution_generator(u0, rhs, seed_time),
            0,
            self.seed_steps_per_step * seed_steps + 1,
            self.seed_steps_per_step,
        ):
            yield u
            us.append(u)

        for t in time.array[seed_steps:-1]:
            u = self.update(t, us, rhs, time.spacing)
            yield u
            us.append(u)


BDF1 = ImplicitEuler


class BDF2(BackwardDifferentiationAbstract):
    """Backward Differentiation of order 2."""

    @property
    def name(self):
        return f"BDF2 (seed: {self.seed.name})"

    @property
    def order(self):
        return 2

    @property
    def us_coeffs(self) -> list[float]:
        return [4 / 3, -1 / 3]

    @property
    def f_coeff(self) -> float:
        return 2 / 3


class BDF3(BackwardDifferentiationAbstract):
    """Backward Differentiation of order 3."""

    @property
    def name(self):
        return f"BDF3 (seed: {self.seed.name})"

    @property
    def order(self):
        return 3

    @property
    def us_coeffs(self) -> list[float]:
        return [18 / 11, -9 / 11, 2 / 11]

    @property
    def f_coeff(self) -> float:
        return 6 / 11


class BDF4(BackwardDifferentiationAbstract):
    """Backward Differentiation of order 4."""

    @property
    def name(self):
        return f"BDF4 (seed: {self.seed.name})"

    @property
    def order(self):
        return 4

    @property
    def us_coeffs(self) -> list[float]:
        return [48 / 25, -36 / 25, 16 / 25, -3 / 25]

    @property
    def f_coeff(self) -> float:
        return 12 / 25


class BDF5(BackwardDifferentiationAbstract):
    """Backward Differentiation of order 5."""

    @property
    def name(self):
        return f"BDF5 (seed: {self.seed.name})"

    @property
    def order(self):
        return 5

    @property
    def us_coeffs(self) -> list[float]:
        return [300 / 137, -300 / 137, 200 / 137, -75 / 137, 12 / 137]

    @property
    def f_coeff(self) -> float:
        return 60 / 137


class BDF6(BackwardDifferentiationAbstract):
    @property
    def name(self):
        return f"BDF6 (seed: {self.seed.name})"

    @property
    def order(self):
        return 6

    @property
    def us_coeffs(self) -> list[float]:
        return [360 / 147, -450 / 147, 400 / 147, -225 / 147, 72 / 147, -10 / 147]

    @property
    def f_coeff(self) -> float:
        return 60 / 147
