"""
Solve the second order equation
x'' = -x
x(0) = 0
x'(0) = 1

which has the solution
x(t) = sin(t)
"""
import matplotlib.pyplot as plt
import numpy as np
from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4

x0 = 0
y0 = 1
A = np.array(
    [
        [0, 1],
        [-1, 0],
    ]
)
u0 = np.array([x0, y0])


def rhs(t, u):
    return A @ u


def exact(t):  # exact solution for testing
    return np.sin(t)


t0, tf = 0, 2 * np.pi
max_time_step = 1e-2
# Create a TimeDomain object
time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, max_time_step)
# Choose a solver
solver = RK4()
plt.ion()  # requires pyqt5 for interactive plotting
for t, u in zip(time.array, solver.solution_generator(u0, rhs, time)):
    # do whatever you want with the solution
    x, y = u
    plt.plot(t, x, "k.")
    plt.pause(1e-3)
plt.show(block=True)
