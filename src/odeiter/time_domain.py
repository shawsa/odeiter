"""A collection of time-discretizations required for odeiter solvers."""
import math
import numpy as np


class TimeDomain:
    """
    An iterable class representing the discretization of a temporal domain.
    This is used as an input to the `odeiter.time_integrator.TimeInterator` solvers.
    """

    def __init__(self, start: float, spacing: float, steps: int):
        """
        A discretization of the interval $[\\text{start},\\  \\text{start} +
        \\text{spacing}\\cdot\\text{steps}]$.

        Parameters:
            start: The inital time.
            spacing: The space between time-steps.
            steps: The total number of time steps.

        Examples:
            >>> t0 = 0
            >>> dt = 0.1
            >>> steps = 5
            >>> time = TimeDomain(t0, dt, steps)
            >>> print(time.array)
            [0.  0.1 0.2 0.3 0.4 0.5]
        """
        self.start = start
        self.spacing = spacing
        self.steps = steps
        self.initialze_array()

    def initialze_array(self):
        self.array = self.start + self.spacing * np.arange(self.steps + 1, dtype=float)

    def __iter__(self):
        yield from self.array


class TimeDomain_Start_Stop_MaxSpacing(TimeDomain):
    def __init__(self, start: float, stop: float, max_spacing: float):
        """An iterable discretization of the inverval $[\\text{start},\\ \\text{stop}]$
        with a spacing of `max_spacing` or smaller.

        Parameters:
            start: The initial time.
            stop: The final time.
            max_spacing: an upper bound on the temporal step-size.

        Examples:
            >>> t0 = 0
            >>> tf = 0.5
            >>> max_dt = 0.11
            >>> time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, max_dt)
            >>> print(time.array)
            [0.  0.1 0.2 0.3 0.4 0.5]
        """
        self.start = start
        self.steps = math.ceil((stop - start) / max_spacing)
        self.spacing = (stop - start) / self.steps
        self.initialze_array()


class TimeDomain_Start_Stop_Steps(TimeDomain):
    def __init__(self, start: float, stop: float, steps: int):
        """An iterable discretization of the inverval $[\\text{start},\\ \\text{stop}]$
        with $\\text{steps} + 1$ equally spaced ponits.

        Parameters:
            start: The initial time.
            stop: The final time.
            steps: The number of time steps.

        Examples:
            >>> t0 = 0
            >>> tf = 0.5
            >>> steps = 5
            >>> time = TimeDomain_Start_Stop_Steps(t0, tf, steps)
            >>> print(time.array)
            [0.  0.1 0.2 0.3 0.4 0.5]
        """
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
    A variant of Timedomain that has no end time.

    Only use this with odeiter.TimeIntegrator.solution_generator.
    This is effectively a while-loop, so always program a termination condition.

    Do not use this with odeiter.TimeIntegrator.solve or odeiter.TimeIntgegraor.t_final.
    Doing so will resut in an infinite loop.

    This is useful for simulating a system into the future for an amount of time that is
    unkown from the start. For example, simulating until the difference between two
    solutions is above a threshold.
    """

    def __init__(self, start: float, spacing: float):
        """An iterable discretization of the inverval $[\\text{start},\\ \\infty)$
        with with `spacing` space between points.

        Parameters:
            start: The initial time.
            spacing: The space between time-steps.

        Examples:
            >>> t0 = 0
            >>> spacing = 0.1
            >>> time = TimeRay(t0, spacing)
            >>> for t, _ in zip(time, range(6)):
            ...     print(t)
            0.0
            0.1
            0.2
            0.30000000000000004
            0.4
            0.5
        """
        self.start = start
        self.spacing = spacing

    @property
    def array(self):
        return Ray(self.start, self.spacing)

    def __iter__(self):
        return self.array.__iter__()
