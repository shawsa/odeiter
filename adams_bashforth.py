from .time_domain import TimeDomain, TimeDomain_Start_Stop_Steps, TimeRay
from .time_integrator import TimeIntegrator
from abc import ABC, abstractmethod, abstractproperty
from collections import deque
from itertools import islice


class AdamsBashforthAbstract(TimeIntegrator):
    def __init__(self, seed: TimeIntegrator, seed_steps_per_step):
        self.seed = seed
        self.seed_steps_per_step = seed_steps_per_step
        self.seed_steps = self.order - 1

    @abstractproperty
    def name(self) -> str:
        ...
        return f"AB4 (seed: {self.seed.name})"

    @abstractmethod
    def update(self, u, fs, delta_t):
        ...

    def solution_generator(self, u0, rhs, time: TimeDomain):
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

        t = time.start + time.spacing * seed_steps
        for t in time.array[seed_steps:-1]:
            u = self.update(u, fs, time.spacing)
            yield u
            fs.append(rhs(t, u))


class AB2(AdamsBashforthAbstract):
    @property
    def order(self):
        return 2

    @property
    def name(self):
        return f"AB2 (seed: {self.seed.name})"

    def update(self, u, fs, delta_t):
        return u + delta_t * (3 / 2 * fs[-1] - 1 / 2 * fs[-2])


class AB4(AdamsBashforthAbstract):
    @property
    def order(self):
        return 4

    @property
    def name(self):
        return f"AB4 (seed: {self.seed.name})"

    def update(self, u, fs, delta_t):
        return u + delta_t * (
            55 / 24 * fs[-1] - 59 / 24 * fs[-2] + 37 / 24 * fs[-3] - 9 / 24 * fs[-4]
        )
