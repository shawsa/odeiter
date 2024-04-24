from scipy.optimize import root
from .single_step import SingleStepMethod


def default_root_finder(fun, x0):
    sol = root(fun=fun, x0=x0)
    if not sol.success:
        raise ValueError(sol)
    return sol.x


class Trapezoidal(SingleStepMethod):

    def __init__(self, root_finder=default_root_finder):
        self.root_finder = root_finder

    @property
    def order(self):
        return 2

    @property
    def name(self):
        return "Trapezoidal"

    def update(self, t, u, f, h):
        fn = f(t, u)

        def fun(x):
            return -x + u + h/2 * (fn + f(t, x))

        return self.root_finder(fun, u)
