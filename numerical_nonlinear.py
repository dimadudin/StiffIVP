from numpy.linalg import solve, norm
import numpy as np

def fixedpoint(g, x0, tol=1.e-3, maxit=10):
    x = x0
    # for _ in range(maxit):
    #     x_old = x
    #     x = g(x)
    #     if abs(x-x_old) < tol: break
    return x

def newton(f, j, x0, tol = 1.e-3, maxit=10):
    x = x0
    for _ in range(maxit):
        fx = f(x)
        if norm(fx, np.inf) < tol: break
        Jx = j(x)
        delta = solve(Jx, -fx) 
        x = x + delta            
    return x


def constJnewton(F, J, x0, tol = 1.e-3, max_iter=10):
    x = x0
    # Jx = J(x)
    # for _ in range(max_iter):
    #     fx = F(x)
    #     delta = solve(Jx, -fx)
    #     x_old = x
    #     x = x + delta
    #     if abs(x-x_old) < tol: break
    return x