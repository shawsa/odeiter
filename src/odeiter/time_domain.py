"""A collection of time-discretizations required for odeiter solvers."""
import math
import numpy as np


class TimeDomain:
    """
    A class representing the discretization of a temporal domain.
    This is used as an input to the odeiter.time_integrator.TimeInterator solvers.

    TimeDomain(start: float, spacing: float, steps: int)

    Represents a discretization of the time interval [start, spacing*steps]
    with steps+1 points including endpoints.

    >>>t0 = 0
    >>>dt = 0.1
    >>>steps = 5
    >>>time = TimeDomain(t0, dt, steps)
    >>>print(time.array)
    """

    def __init__(self, start: float, spacing: float, steps: int):
        self.start = start
        self.spacing = spacing
        self.steps = steps
        self.initialze_array()

    def initialze_array(self):
        self.array = self.start + self.spacing * np.arange(self.steps + 1, dtype=float)

    def __iter__(self):
        yield from self.array


class TimeDomain_Start_Stop_MaxSpacing(TimeDomain):
    """
    A variant of odeiter.Timedomain that accepts different parameters.
    TimeDomain_Start_Stop_MaxSpacing(start: float, stop: float, max_spacing: int)

    Discretizes the temporal interval [start, stop] with stepsize less than max_spacing.
    """

    def __init__(self, start: float, stop: float, max_spacing: float):
        self.start = start
        self.steps = math.ceil((stop - start) / max_spacing)
        self.spacing = (stop - start) / self.steps
        self.initialze_array()


class TimeDomain_Start_Stop_Steps(TimeDomain):
    """
    A variant of odeiter.Timedomain that accepts different parameters.
    TimeDomain_Start_Stop_Steps(start: float, stop: float, max_spacing: int)

    Discretizes the temporal interval [start, stop] with `steps` equally sized steps.
    """

    def __init__(self, start: float, stop: float, steps: int):
        self.start = start
        self.steps = steps
        self.spacing = (stop - start) / steps
        self.initialze_array()


class Ray:
    def __init__(self, start: float, step: float):
        self.start = start
        self.step = step

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.start + key * self.step
        if not isinstance(key, slice):
            raise ValueError
        start = key.start or 0
        step = key.step or 1
        return Ray(self[start], self.step * step)

    def __iter__(self):
        index = 0
        while True:
            yield self.start + index * self.step
            index += 1

    def __repr__(self):
        return f"Ray(start={self.start}, step={self.step})"


class TimeRay(TimeDomain):
    """
    A variant of odeiter.Timedomain that has no end time.

    Only use this with odeiter.TimeIntegrator.solution_generator.
    This is effectively a while-loop, so always program a termination condition.

    Do not use this with odeiter.TimeIntegrator.solve or odeiter.TimeIntgegraor.t_final.
    Doing so will resut in an infinite loop.

    TimeRay(start: float, spacing: float)

    Discretizes the temporal interval [start, oo) with `steps` equally sized steps.

    This is useful for simulating a system into the future for an amount of time that is
    unkown from the start. For example, simulating until the difference between two
    solutions is above a threshold.
    """

    def __init__(self, start: float, spacing: float):
        self.start = start
        self.spacing = spacing

    @property
    def array(self):
        return Ray(self.start, self.spacing)

    def __iter__(self):
        return self.array.__iter__()
