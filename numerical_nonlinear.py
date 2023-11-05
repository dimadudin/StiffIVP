from numpy.linalg import norm, solve
import numpy as np

def newton(F, J, x0, tol = 1.e-10, max_iter=20):
    x = x0
    for _ in range(max_iter):
        fx = F(x)
        if norm(fx, np.inf) < tol: break
        Jx = J(x)
        delta = solve(Jx, -fx) 
        x = x + delta            
    return x