import numpy as np
import matplotlib.pyplot as plt
from ivp import solveIVP


if __name__ == 'main':
    def f_test(t,y):
      return np.exp(y)*(1+t)
    
    def f_brunner(t,y):
        return [-0.013*y[1] - 1e3*y[0]*y[1] - 2.5e3*y[0]*y[2],
                -0.013*y[1] - 1e3*y[0]*y[1],
                -2.5e3*y[0]*y[2]]

    def f_robertson(t,y):
        return [-0.04*y[0] + 1e4*y[1]*y[2],
                0.04*y[0] - 1e4*y[1]*y[2] - 3e7*y[1]*y[1],
                3e7*y[1]*y[1]]

    sol = solveIVP(f_test, [0, 1], [-0.5], method='RK')
    # sol = solveIVP(f_robertson, [0, 5], [1,0,0], method='RK')
    plt.plot(sol.t, sol.y.T)
    plt.show()