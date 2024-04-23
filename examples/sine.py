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
from odeiter import TimeDomain_Start_Stop_MaxSpacing, Euler, AB2, RK4, AB4
from tqdm import tqdm


# define the system and parameters
x0 = 0
y0 = 1
t0, tf = 0, 3.5 * np.pi

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


# simple example
max_time_step = 1e-3
time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, max_time_step)
# solver = RK4()
solver = AB4(seed=RK4(), seed_steps=1)
xs = np.array([u[0] for u in solver.solve(u0, rhs, time)])
fig, (ax_sol, ax_err) = plt.subplots(2, 1, sharex=True)
ax_sol.plot(time.array, xs, label=solver.name)
ax_sol.plot(time.array, exact(time.array), "--", linewidth=3, label="True")
ax_err.semilogy(time.array, np.abs(xs - exact(time.array)))
ax_sol.legend(loc="lower left")
ax_err.legend(loc="lower left")
ax_err.set_xlabel("$t$")
ax_err.set_ylabel("error")


# Multiple simultainous solves
max_time_step = 3e-2
time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, max_time_step)
solvers = [
    Euler(),
    AB2(seed=Euler(), seed_steps=2),
    RK4(),
    AB4(seed=RK4(), seed_steps=1),
]
colors = ["blue", "green", "red"]

plt.ion()
fig, (ax_sol, ax_err) = plt.subplots(2, 1, sharex=True)
ax_sol.plot(time.array, exact(time.array), "k-", label="exact")

for solver, color in zip(solvers, colors):
    ax_sol.plot([], [], marker=".", color=color, label=f"{solver.name}")
    ax_err.semilogy([], [], marker=".", color=color, label=f"{solver.name}")

ax_sol.legend(loc="lower left")
ax_err.legend(loc="lower left")
ax_err.set_xlabel("$t$")
ax_err.set_ylabel("error")

for t, *us in zip(
    time.array, *(solver.solution_generator(u0, rhs, time) for solver in solvers)
):
    for color, (x, _) in zip(colors, us):
        ax_sol.plot(t, x, color=color, marker=".")
        x_true = exact(t)
        err = abs(x - x_true)
        ax_err.semilogy(t, err, color=color, marker=".")

    plt.pause(1e-3)


# convergence plot
x_true = exact(tf)
time_steps = np.logspace(-3, -1, 10)
plt.figure("Convergence")
for solver, order in [
    (Euler(), 1),
    (AB2(seed=Euler(), seed_steps=2), 2),
    (RK4(), 4),
    (AB4(seed=RK4(), seed_steps=1), 4),
]:
    errs = []
    for k in tqdm(time_steps):
        time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, k)
        u = solver.t_final(u0, rhs, time)
        x = u[0]
        err = abs((x - x_true) / x_true)
        errs.append(err)
    plt.loglog(time_steps, errs, "o", label=solver.name)
    order_err = errs[-1] * np.exp(order * np.log(time_steps[0] / time_steps[-1]))
    plt.plot(
        [time_steps[0], time_steps[-1]],
        [order_err, errs[-1]],
        label=f"$\\mathcal{{O}}({order})$",
    )

plt.xlabel("Time Step")
plt.ylabel("Relative Error")
plt.title("Convergence")
plt.legend()
plt.grid()