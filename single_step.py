from abc import abstractmethod
from .time_domain import TimeDomain
from .time_integrator import TimeIntegrator


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
    def order(self):
        return 1

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
    def order(self):
        return 1

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

    @property
    def order(self):
        return 4

    def update(self, t, u, f, h):
        k1 = f(t, u)
        k2 = f(t + h / 2, u + h / 2 * k1)
        k3 = f(t + h / 2, u + h / 2 * k2)
        k4 = f(t + h, u + h * k3)
        return u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
