import numpy as np
from odebase import ODESolver

def step(fun, t, y, f, h, A, B, C, K):
    K[0] = f
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dy = np.dot(K[:s].T, a[:s]) * h
        K[s] = fun(t + c * h, y + dy)
    y_new = y + h * np.dot(K[:-1].T, B)
    f_new = fun(t + h, y_new)
    K[-1] = f_new
    return y_new, f_new


class RungeKutta(ODESolver):
    C: np.ndarray = NotImplemented
    A: np.ndarray = NotImplemented
    B: np.ndarray = NotImplemented
    order: int = NotImplemented
    stages: int = NotImplemented

    def __init__(self, fun, t0, tn, y0, rtol=1e-3, atol=1e-6, h=1e-3):
        super().__init__(fun, t0, tn, y0)
        self.rtol, self.atol = rtol, atol
        self.f = self.fun(self.t, self.y)
        self.h = h
        self.K = np.empty((self.stages + 1, self.n), dtype=self.y.dtype)

    def _step_impl(self):
        t = self.t
        y = self.y

        h = self.h
        while 1:
            t_ = t + h
            y_, f_ = step(self.fun, t, y, self.f, h, self.A, self.B, self.C, self.K)

        self.t = t_
        self.y = y_

        self.h = h
        self.f = f_

        return True, None

class RK(RungeKutta):
    order = 3
    stages = 3
    C = np.array([0, 1/2, 1])
    A = np.array([
        [0, 0, 0],
        [1/2, 0, 0],
        [-1, 2, 0]])
    B = np.array([1/6, 2/3, 1/6])