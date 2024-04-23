import matplotlib.pyplot as plt
import numpy as np
from odeiter import TimeDomain_Start_Stop_MaxSpacing, TqdmWrapper, Euler, AB2, RK4, AB4
from tqdm import tqdm


# define the system and parameters
x0 = 0
y0 = 1
t0, tf = 0, 15

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


# Solution
max_time_step = 3e-2
time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, max_time_step)
# solver = Euler()
# solver = AB2(seed=RK4(), seed_steps_per_step=1)
solver = AB4(seed=RK4(), seed_steps_per_step=1)
# solver = AB4(AB2(seed=Euler(), seed_steps_per_step=2), seed_steps_per_step=2)
color = "blue"

plt.ion()
fig, (ax_sol, ax_err) = plt.subplots(2, 1, sharex=True)
ax_sol.plot(time.array, exact(time.array), "k-", label="exact")

ax_sol.plot([], [], marker=".", color=color, label=f"{solver.name}")
ax_err.semilogy([], [], marker=".", color=color, label=f"{solver.name}")

ax_sol.legend(loc="lower left")
ax_err.legend(loc="lower left")
ax_err.set_xlabel("$t$")
ax_err.set_ylabel("error")

for t, u in zip(time.array, TqdmWrapper(solver).solution_generator(u0, rhs, time)):
    pass
    x = u[0]
    ax_sol.plot(t, x, color=color, marker=".")
    x_true = exact(t)
    err = max(1e-16, abs(x - x_true))
    ax_err.semilogy(t, err, color=color, marker=".")

    plt.pause(1e-3)


# convergence plot
x_true = exact(tf)
time_steps = np.logspace(-3, -1, 10)
plt.figure("Convergence")
errs = []
for k in tqdm(time_steps):
    time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, k)
    u = solver.t_final(u0, rhs, time)
    x = u[0]
    err = abs((x - x_true) / x_true)
    errs.append(err)
plt.loglog(time_steps, errs, "o", label=solver.name)

for order in range(1, 5):
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
