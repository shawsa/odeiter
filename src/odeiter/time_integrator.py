"""A time-integration class for solving ODEs numerically."""
from .time_domain import TimeDomain

from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from tqdm import tqdm
from typing import Callable, Generator


class TimeIntegrator(ABC):
    @abstractproperty
    def name(self) -> str:
        ...

    @abstractproperty
    def order(self) -> int:
        """The order of the method. If this method is order o
        then cutting the time-step in half should reduce the error
        by a factor of (1/2)^o.

        This is limited by machine precision. Additionally, it may not exhibit
        this convergence behaviour if the time step is too large, or if the
        forcing term is not smooth enough.
        """
        ...

    @abstractmethod
    def solution_generator(
        self,
        u0: np.ndarray[float],
        rhs: Callable[[float, np.ndarray[float]], np.ndarray[float]],
        time: TimeDomain,
    ) -> Generator[np.ndarray[float], None, None]:
        """
        Return a generator that yields the solution at each time in time.array.

        u0: a numpy array of initial conditions. It should have the same shape as
            the solution at each time step.

        rhs: A function rhs(t, u) that is the right-hand-side of the equation
             u' = rhs(t, u).

        time: An odeiter.TimeDomain instance.

        Returns: a generator that yeilds the solution at each time
        step in time.array.
        """
        ...

    def solve(
        self,
        u0: np.ndarray[float],
        rhs: Callable[[float, np.ndarray[float]], np.ndarray[float]],
        time: TimeDomain,
    ) -> list[np.ndarray[float]]:
        """
        Return a list of the solutions at each time for times in time.array.
        Equivalent to list(solver.solution_generator(u0, rhs, time)).
        See TimeIntegrator.solution_generator for parameter inputs.
        """
        return list(self.solution_generator(u0, rhs, time))

    def t_final(
        self,
        u0: np.ndarray[float],
        rhs: Callable[[float, np.ndarray[float]], np.ndarray[float]],
        time: TimeDomain,
    ) -> list[np.ndarray[float]]:
        """
        Returns the solution at the final time time.array[-1].
        Equivalent to solver.solve[-1].
        See TimeIntegrator.solution_generator for parameter inputs.
        """
        for u in self.solution_generator(u0, rhs, time):
            pass
        return u


class TqdmWrapper:
    def __init__(self, solver: TimeIntegrator):
        self.solver = solver

    def solve(self, u0, rhs, time: TimeDomain):
        return list(
            tqdm(self.solver.solution_generator(u0, rhs, time), total=time.steps)
        )

    def t_final(self, u0, rhs, time: TimeDomain):
        for u in tqdm(self.solver.solution_generator(u0, rhs, time), total=time.steps):
            pass
        return u

    def solution_generator(self, u0, rhs, time: TimeDomain):
        yield from tqdm(self.solver.solution_generator(u0, rhs, time), total=time.steps)
