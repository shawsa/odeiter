"""A module for solving systems of differential equations.
Specifically inital value problems of the form
    u'(t) = rhs(t, u)
    u(0) = u0
To sovle this with odeiter you will need a discretized time domain from odeiter.time_domain
and a solver that subclasses odeiter.time_integrator.

>>>from odeiter import TimeDomain, RK4
>>>time = TimeDomain(0, 0.1, 10)  # discretize the interval [0, 1] with 11 points
>>>solver = RK4()
>>>for u in solver.solution_generator(u0, rhs, time):
>>>    # do something with the soluiton at each step
>>>    print(u)
"""
from .time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
from .time_integrator import TqdmWrapper
from .single_step import Euler, Trapezoidal, RK4
