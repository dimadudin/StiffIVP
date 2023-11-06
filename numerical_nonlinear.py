from numpy.linalg import solve
import numpy as np

def fixedpoint(g, x0, tol=1.e-8, max_iter=30):
    x = x0
    for _ in range(max_iter):
        x_old = x
        x = g(x)
        # if abs(x-x_old) < tol: break
    return x

def newton(F, J, x0, tol = 1.e-8, max_iter=30):
    x = x0
    for _ in range(max_iter):
        fx = F(x)
        Jx = J(x)
        delta = solve(Jx, -fx)
        # x_old = x
        # x = x + delta
        # if abs(x-x_old) < tol: break
    return x


def constJnewton(F, J, x0, tol = 1.e-8, max_iter=30):
    x = x0
    Jx = J(x)
    for _ in range(max_iter):
        fx = F(x)
        delta = solve(Jx, -fx)
        # x_old = x
        # x = x + delta
        # if abs(x-x_old) < tol: break
        return x