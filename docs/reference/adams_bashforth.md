Adams-Bashforth solvers are a family of explicit solvers that perform a single
right-hand-side evaluation per time step and achieve high-order accuracy by
storing the derivative at several previous time steps. This makes them
computationally efficent at the cost of some memory.

Adams-Bashforth methods are mult-step solvers and thus
require not only an initial condition, but also the solution at
the first few time points dependeing on the order of the method.
For example, AB3 requires the solution at times $t_0, t_0+k, t_0+2k$
where $t_0$ is the inital time and $k$ is the temporal step-size.

This implementation assumes that you only have the solution at $t_0$
and accepts another time-integrator as a seed. For example, we may use
[RK4](single_step.md) (a single step method). You must also decide the
step-size of the single-step method such that the step-size of the
Adams-Bashfort solver is an integer multiple.

For example,
```
	solver = AB5(RK4(), seed_steps_per_step=2)
```
Will use the initial condition and RK4 to take 8 time times steps
each at half the step size. This will generate solutions at times
$t_0, t_0 + k/2, t_0 + k, t0+ 3k/2, ..., t_0 + 4k$.
It will then subsample these as necessary to use as appropriate seed
steps.

It is important to ensure that the order of the seed method and the number
of seed steps per step are at least as accurate as the orer of the
Adams-Bashforth method. For example
```
solver = AB5(RK4(), seed_steps_per_step=1)
```
would not give a fifth order method because RK4 will generate seed steps
up to fourth order accuracy. In contrast
```
solver = AB5(RK4(), seed_steps_per_step=2)
```
will generate seed steps to eighth order accuracy, relative to the
step size of the Adams-Bashfort solver.

# adams_bashforth
::: odeiter.adams_bashforth
