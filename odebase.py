class ODESolver:
    """Base class for iterative ODE solvers"""
    def __init__(self, f, t0, tn, y0):
        self.f = f
        self.t, self.y = t0, y0
        self.tn = tn
        self.m = self.y.size
        self.status = 'run'

    def step(self):
        self.status = self._step_impl()
        return self.status

    def _step_impl(self):
        raise NotImplementedError