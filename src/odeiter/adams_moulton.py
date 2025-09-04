"""A module containing Adams-Moulton solvers of several orders."""
from .single_step import Trapezoidal
from .root_finder import RootFinder, DefaultRootFinder
from .time_domain import TimeDomain, TimeRay
from .time_integrator import TimeIntegrator
from abc import abstractproperty
from collections import deque
from itertools import islice
import numpy as np
from typing import Callable, Generator


class AdamsMoultonAbstract(TimeIntegrator):
    """
    An abstract class for Adams-Moulton (AM) solvers.
    All AM solvers will inherit these methods. Additionally,
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
        self.seed_steps = self.order - 2
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
    def fs_coeffs(self) -> list[float]:
        """
        Returns:
            The interpolation coefficients of the method.
        """
        ...

    def update(
        self,
        t: float,
        u: np.ndarray[float],
        rhs: Callable[[float, np.ndarray[float]], np.ndarray[float]],
        fs: np.ndarray[float],
        delta_t: float,
    ) -> np.ndarray[float]:
        """
        Compute the next time step. You probably want `solution_generator` instead.

        Parameters:
            t: The current time.
            u: The solution at the current time-step.
            rhs: The right-hand-side of the system as a function `rhs(t, u) -> u'`.
            fs: the right-hand-side at several previous time-steps.
            delta_t: the temporal step-size.

        Returns:
            The solution at the next time step.
        """
        const_vec = u + delta_t * sum(
            c * f for c, f in zip(self.fs_coeffs[-1:0:-1], fs)
        )

        def func(x):
            return -x + const_vec + delta_t * self.fs_coeffs[0] * rhs(t + delta_t, x)

        return self.root_finder(func, u)

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
        fs = deque(maxlen=self.order - 1)
        for u in islice(
            self.seed.solution_generator(u0, rhs, seed_time),
            0,
            self.seed_steps_per_step * seed_steps + 1,
            self.seed_steps_per_step,
        ):
            yield u
            fs.append(rhs(t, u))
            t += time.spacing

        for t in time.array[seed_steps:-1]:
            u = self.update(t, u, rhs, fs, time.spacing)
            yield u
            fs.append(rhs(t + time.spacing, u))


AM1 = Trapezoidal


class AM2(AdamsMoultonAbstract):
    """Adams Moulton of order 3."""

    @property
    def order(self):
        return 3

    @property
    def name(self):
        return f"AM2 (seed: {self.seed.name})"

    @property
    def fs_coeffs(self) -> list[float]:
        return [5 / 12, 8 / 12, -1 / 12]


class AM3(AdamsMoultonAbstract):
    """Adams Moulton of order 4."""

    @property
    def order(self):
        return 4

    @property
    def name(self):
        return f"AM3 (seed: {self.seed.name})"

    @property
    def fs_coeffs(self) -> list[float]:
        return [9 / 24, 19 / 24, -5 / 24, 1 / 24]


class AM4(AdamsMoultonAbstract):
    """Adams Moulton of order 5."""

    @property
    def order(self):
        return 5

    @property
    def name(self):
        return f"AM4 (seed: {self.seed.name})"

    @property
    def fs_coeffs(self) -> list[float]:
        return [251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720]
