import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import numpy as np
from odeiter import TimeDomain_Start_Stop_MaxSpacing, Euler, RK4, Trapezoidal
from odeiter.backward_differentiation import BDF3
from odeiter.adams_bashforth import AB5
from odeiter.adams_moulton import AM4
import os

PROBLEM = """
Solve the second order equation
x'' = -x
x(0) = 0
x'(0) = 1

which has the solution
x(t) = sin(t)
"""

FILE_NAME = "simultaneous_solves.gif"

x0 = 0
y0 = 1
t0, tf = 0, 5

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


solvers = [
    Euler(),
    Trapezoidal(),
    BDF3(seed=RK4(), seed_steps_per_step=1),
    RK4(),
    AB5(seed=RK4(), seed_steps_per_step=2),
    AM4(seed=RK4(), seed_steps_per_step=2),
]

# Multiple simultainous solves
max_time_step = 3e-2
time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, max_time_step)
colors = TABLEAU_COLORS.values()

plt.ion()
fig, (ax_sol, ax_err) = plt.subplots(2, 1, sharex=True)
ax_sol.plot(time.array, exact(time.array), "k-", label="exact")

for solver, color in zip(solvers, colors):
    ax_sol.plot([], [], marker=".", color=color, label=f"{solver.name}")
    ax_err.semilogy([], [], marker=".", color=color, label=f"{solver.name}")

for ax in [ax_sol, ax_err]:
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax_err.set_xlabel("$t$")
ax_err.set_ylabel("error")
plt.tight_layout()

with imageio.get_writer(FILE_NAME, mode="I", loop=0, duration=0.05) as writer:
    for t, *us in zip(
        time.array, *(solver.solution_generator(u0, rhs, time) for solver in solvers)
    ):
        for color, (x, _) in zip(colors, us):
            ax_sol.plot(t, x, color=color, marker=".")
            x_true = exact(t)
            err = abs(x - x_true)
            ax_err.semilogy(t, err, color=color, marker=".")
        plt.savefig(FILE_NAME + ".png")
        image = imageio.imread(FILE_NAME + ".png")
        writer.append_data(image)
        plt.pause(1e-3)

os.remove(FILE_NAME + ".png")
plt.show()
plt.close()
