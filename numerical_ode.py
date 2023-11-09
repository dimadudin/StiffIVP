# Реализация методов Рунге-Кутты для решения ОДУ #
import numpy as np
from numerical_nonlinear import newton
from numpy.linalg import det
import matplotlib.pyplot as plt

# Явный метод Рунге-Кутты #
class explicit_rk:
    def __init__(self, a, b, c):
        # Таблица Бутчера #
        self.a, self.b, self.c = a, b, c
    def explicit(self, tj, yj, f, dt, s):
        a, b, c = self.a, self.b, self.c
        # Смещение стадийной касательной #
        k = np.array([np.zeros_like(yj, dtype=np.float64) for _ in range(s)], dtype=np.float64)
        # Смещение приближенного значения #
        dy = np.zeros_like(yj, dtype=np.float64)
        # Итерация по стадиям (i) #
        for i in range(s):
            zi = np.zeros_like(yj, dtype=np.float64)
            # Итерация по предыдущим (т.к. явный метод) стадиям (l) #
            for l in range(i):
                zi += dt * a[i,l] * k[l]
            k[i] = f(tj + dt * c[i],yj + zi)
            dy += dt * b[i] * k[i]
        return dy

    def __call__(self, f, y0, t0, T, n):
        # Таблица Бутчера #
        a, b, c = self.a, self.b, self.c
        # Временная сетка #
        t, dt = np.linspace(t0, T, n, retstep=True)
        # Приближенные значения в узлах сетки #
        y = [y0]
        # Количество стадий #
        s = len(b)
        # Итерация по временной сетке (j) #
        for j in range(n-1):
            dy = self.explicit(t[j], y[j], f, dt, s)
            y.append(y[j] + dy)
        return (np.array(t, dtype=np.float64), np.array(y, dtype=np.float64))


# Неявный метод Рунге-Кутты #
class implicit_rk:
    def __init__(self, a, b, c):
        # Таблица Бутчера #
        self.a, self.b, self.c = a, b, c
    def __call__(self, f, df, y0, t0, T, n):
        # Таблица Бутчера #
        a, b, c = self.a, self.b, self.c
        # Временная сетка #
        t, dt = np.linspace(t0, T, n, retstep=True)
        # Приближенные значения в узлах сетки #
        y = [y0]
        # Количество стадий #
        s = len(b)
        # Итерация по временной сетке (j) #
        for j in range(n-1):
            # Смещение стадийной касательной z #
            # Система нелинейных уравнений F(z)=0 #
            def F(z):
                F = np.array([z[i] for i in range(s)], dtype=np.float64)
                # Итерация по стадиям (i) #
                for i in range(s):
                    # Итерация по всем (т.к. неявный метод) стадиям (l) #
                    for l in range(s):
                        F[i] -= dt * a[i,l] * f(t[j] + dt * c[l],y[j] + z[l])
                return F
            # Якобиан F(z) #
            def J(z):
                J = np.identity(s, dtype=np.float64)
                # Итерация по стадиям (i) #
                for i in range(s):
                    # Итерация по всем (т.к. неявный метод) стадиям (l) #
                    for l in range(s):
                        J[i][l] -= dt * a[i,l] * df(t[j] + dt * c[l],y[j] + z[l])[i,l]
                return J
            # Начальное приближение z0 #
            z0 =  np.array([np.zeros_like(y0) for _ in range(s)], dtype=np.float64)
            # Метод Ньютона #
            z = newton(F, J, z0)
            # Смещение приближенного значения #
            dy = np.zeros_like(y0, dtype=np.float64)
            # Итерация по стадиям (i) #
            for i in range(s):
                dy += dt * b[i] * f(t[j] + dt * c[i],y[j] + z[i])
            y.append(y[j] + dy)
        return (np.array(t, dtype=np.float64), np.array(y, dtype=np.float64))
    
def plot_R(a, b):
    x = np.linspace(-3,3,100)
    y = np.linspace(-3,3,100)
    X,Y = np.meshgrid(x,y)

    R = np.identity(len(x), dtype=np.float64)
    P = np.identity(len(x), dtype=np.float64)

    for i in range(len(x)):
        for j in range(len(y)):
            E = np.identity(len(a), dtype=np.float64)
            EzA = E - (x[i]+y[j]*1j) * a
            delta = det(EzA)
            delta1 = det(EzA + (x[i]+y[j]*1j)
                            * np.outer(np.ones(len(b)), b))
            P[j][i] = delta1/delta
            R[j][i] = abs(delta1/delta)

    plt.figure()
    plt.plot(P)
    plt.grid()
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.show()

    plt.figure()
    plt.contourf(X,Y,R, 1, levels = [0,1])
    plt.contour(X,Y,R, 1, colors = 'black', levels = [1])

    plt.plot([min(x),max(x)],[0,0], 'k--')
    plt.plot([0,0],[min(y),max(y)], 'k--')
 
    plt.grid()
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.show()