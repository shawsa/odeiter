## Basic syntax
For example, to solve the initial value problem
$$
\\begin{align}
	u''(t) &= -u \\\\
	u(0) &= 0 \\\\
	u'(0) &= 1
\\end{align}
$$
we can express it as a first-order system
$$
\\begin{align}
	\\vec{u}' &= \\begin{bmatrix} 0 & 1 \\\\ -1 & 0 \\end{bmatrix} \\vec{u} \\\\
	\\vec{u} &= \\begin{bmatrix}0 \\\\ 1 \\end{bmatrix}
\\end{align}
$$
then solve the system with a time-integrator like [RK4](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) like so:

```python
import numpy as np
from odeiter import TimeDomain, RK4
u0 = np.array([0.0, 1.0])

def rhs(t, u):
    return np.array([[0, 1], [-1, 0]]) @ u

time = TimeDomain(0, 0.1, 4)
solver = RK4()
for u in solver.solution_generator(u0, rhs, time):
    # so something with the solution at this time
    print(u)

# OUTPUT
# [0. 1.]
# [0.09983333 0.99500417]
# [0.19866917 0.9800666 ]
# [0.29551996 0.95533654]
# [0.38941803 0.9210611 ]
```

# Interactive Plotting

This shows some of the flexibily by solving the system above, plotting
the solution and plotting the error.

The plotting occurs as the system is solved, rather than after it is solved
and requires `pyqt5` and `matplotlib` to function properly.

```python
--8<-- "basic_example.py"
```
