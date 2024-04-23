"""A time-integration class for solving ODEs numerically."""
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import numpy.linalg as la
from .time_domain import TimeDomain


class TimeIntegrator(ABC):
    @abstractproperty
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def solution_generator(self, u0, rhs, time: TimeDomain):
        raise NotImplementedError

    def solve(self, u0, rhs, time: TimeDomain):
        return list(self.solution_generator(u0, rhs, time))

    def t_final(self, u0, rhs, time: TimeDomain):
        for u in self.solution_generator(u0, rhs, time):
            pass
        return u


# solution generator instead.
class SingleStepMethod(TimeIntegrator):
    @abstractmethod
    def update(self, t, u, f, h):
        raise NotImplementedError

    def solution_generator(self, u0, rhs, time: TimeDomain):
        u = u0
        yield u
        for t in time.array[:-1]:
            u = self.update(t, u, rhs, time.spacing)
            yield u


class Euler(SingleStepMethod):
    @property
    def name(self):
        return "Euler"

    def update(self, t, u, f, h):
        return u + h * f(t, u)


class EulerDelta(SingleStepMethod):
    def __init__(self, delta_time, delta_profile):
        self.delta_time = delta_time
        self.delta_profile = delta_profile

    @property
    def name(self):
        return "Euler-Delta"

    def update(self, t, u, f, h):
        u_new = u + h * f(t, u)
        if abs(t - self.delta_time) < h / 2:
            u_new += self.delta_profile
        return u_new


class RK4(SingleStepMethod):
    @property
    def name(self):
        return "RK4"

    def update(self, t, u, f, h):
        k1 = f(t, u)
        k2 = f(t + h / 2, u + h / 2 * k1)
        k3 = f(t + h / 2, u + h / 2 * k2)
        k4 = f(t + h, u + h * k3)
        return u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class TrapezoidalLinear(SingleStepMethod):
    "rhs needs to be a function t -> A(t) for the matrix falued function A"
    @property
    def name(self):
        return "Trapezoidal"

    def update(self, t, u, f, h):
        return la.solve(np.eye(len(u)) - h / 2 * f(t + h), u + h / 2 * f(t) @ u)
