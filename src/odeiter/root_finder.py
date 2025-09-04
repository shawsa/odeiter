from abc import ABC, abstractstaticmethod
import numpy as np
from scipy.optimize import root
from typing import Callable


class RootFinder(ABC):
    """An interface for root-finders used to solve systems in
    implicit methods.
    """

    @abstractstaticmethod
    def solve(
        func: Callable[[np.ndarray[float]], np.ndarray[float]],
        u0: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        Parameters:
            func: A function `f(u)`.
            u0: An approximate solution to `f(u) = 0`.

        Returns:
            The solution `u` to `f(u) = 0`.

        Raises:
            ValueError: The solver fails to find a solution.
        """
        ...


class DefaultRootFinder(RootFinder):
    """A wrapper for [`scipy.optimize.root`](https://docs.scipy.org/doc/scipy/
    reference/generated/scipy.optimize.root.html).
    """

    @staticmethod
    def solve(
        func: Callable[[np.ndarray[float]], np.ndarray[float]],
        u0: np.ndarray[float],
    ):
        sol = root(fun=func, x0=u0)
        if not sol.success:
            raise ValueError(sol)
        return sol.x
