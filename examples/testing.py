import matplotlib.pyplot as plt
import numpy as np
from odeiter import TimeDomain_Start_Stop_MaxSpacing, TqdmWrapper
from odeiter.single_step import ImplicitEuler, Trapezoidal, RK4
from odeiter.adams_bashforth import AB2, AB3, AB4, AB5
from odeiter.backward_differentiation import BDF2, BDF3, BDF4, BDF5, BDF6
from odeiter.adams_moulton import AM2, AM3, AM4
from tqdm import tqdm


# define the system and parameters
A = np.array(
    [
        [0, 1],
        [-1, 0],
    ]
)

x0 = 0
y0 = -1
t0, tf = 0, 15
u0 = np.array([x0, y0])


def rhs(t, u):
    ret = A @ u
    ret[1] += 3 * np.sin(2 * t)
    return ret


def exact(t):  # exact solution for testing
    return np.sin(t) - np.sin(2 * t)


# Solution
# solver = Euler()
# solver = ImplicitEuler()
# solver = Trapezoidal()
# solver = RK4()
# solver = AB2(seed=RK4(), seed_steps_per_step=1)
# solver = AB3(seed=RK4(), seed_steps_per_step=1)
# solver = AB4(seed=RK4(), seed_steps_per_step=1)
solver = AB5(seed=RK4(), seed_steps_per_step=2)
# solver = AM2(seed=RK4(), seed_steps_per_step=1)
# solver = AM3(seed=RK4(), seed_steps_per_step=1)
# solver = AM4(seed=RK4(), seed_steps_per_step=1)
# solver = BDF2(seed=RK4(), seed_steps_per_step=1)
# solver = BDF3(seed=RK4(), seed_steps_per_step=1)
# solver = BDF4(seed=RK4(), seed_steps_per_step=1)
# solver = BDF5(seed=RK4(), seed_steps_per_step=2)
# solver = BDF6(seed=RK4(), seed_steps_per_step=2)

max_time_step = 1e-1
time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, max_time_step)

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
time_steps = np.logspace(-3, -2, 10)
plt.figure("Convergence")
errs = []
for k in tqdm(time_steps):
    time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, k)
    u = solver.t_final(u0, rhs, time)
    x = u[0]
    err = abs((x - x_true) / x_true)
    errs.append(err)
plt.loglog(time_steps, errs, "o", label=solver.name)

for order in range(1, 7):
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
