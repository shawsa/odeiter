"""A module containing Adams-Bashforth solvers of several orders."""
from .time_domain import TimeDomain, TimeRay
from .time_integrator import TimeIntegrator
from abc import abstractproperty
from collections import deque
from itertools import islice
import numpy as np
from typing import Callable, Generator


class AdamsBashforthAbstract(TimeIntegrator):
    """
    An abstract class for Adams-Bashforth (AB) solvers.
    All AB solvers will inherit these methods.
    """

    def __init__(self, seed: TimeIntegrator, seed_steps_per_step: int):
        """
        Parameters:
            seed: another time integrator used to take the first few steps.
            seed_steps_per_step: the number of seed steps taking per step
                of the AB integrator.
        """
        self.seed = seed
        self.seed_steps_per_step = seed_steps_per_step
        self.seed_steps = self.order - 1

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
        self, u: np.ndarray[float], fs: np.ndarray[float], delta_t: float
    ) -> np.ndarray[float]:
        """
        Compute the next time step. You probably want `solution_generator` instead.

        Parameters:
            u: The solution at the current time-step.
            fs: the right-hand-side at several previous time-steps.
            delta_t: the temporal step-size.

        Returns:
            The solution at the next time step.
        """
        return u + delta_t * sum(c * f for c, f in zip(self.fs_coeffs[::-1], fs))

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
        fs = deque(maxlen=self.order)
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
            u = self.update(u, fs, time.spacing)
            yield u
            fs.append(rhs(t + time.spacing, u))


class AB2(AdamsBashforthAbstract):
    """Adams-Bashforth 2."""

    @property
    def order(self):
        return 2

    @property
    def name(self):
        return f"AB2 (seed: {self.seed.name})"

    @property
    def fs_coeffs(self) -> list[float]:
        return [3 / 2, -1 / 2]


class AB3(AdamsBashforthAbstract):
    """Adams-Bashforth 3."""

    @property
    def order(self):
        return 3

    @property
    def name(self):
        return f"AB3 (seed: {self.seed.name})"

    @property
    def fs_coeffs(self) -> list[float]:
        return [23 / 12, -16 / 12, 5 / 12]


class AB4(AdamsBashforthAbstract):
    """Adams-Bashforth 4."""

    @property
    def order(self):
        return 4

    @property
    def name(self):
        return f"AB4 (seed: {self.seed.name})"

    @property
    def fs_coeffs(self) -> list[float]:
        return [55 / 24, -59 / 24, 37 / 24, -9 / 24]


class AB5(AdamsBashforthAbstract):
    """Adams-Bashforth 5."""

    @property
    def order(self):
        return 5

    @property
    def name(self):
        return f"AB5 (seed: {self.seed.name})"

    @property
    def fs_coeffs(self) -> list[float]:
        return [1901 / 720, -2774 / 720, 2616 / 720, -1274 / 720, 251 / 720]
