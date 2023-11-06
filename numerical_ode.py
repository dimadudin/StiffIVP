# Реализация методов Рунге-Кутты для решения ОДУ #
import numpy as np
from numerical_nonlinear import newton

# Явный метод Рунге-Кутты #
class explicit_rk:
    def __init__(self, a, b, c):
        # Таблица Бутчера #
        self.a, self.b, self.c = a, b, c
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
            # Смещение стадийной касательной #
            k = [np.zeros_like(y0) for _ in range(s)]
            # Смещение приближенного значения #
            dy = np.zeros_like(y0)
            # Итерация по стадиям (i) #
            for i in range(s):
                zi = np.zeros_like(y0)
                # Итерация по предыдущим (т.к. явный метод) стадиям (l) #
                for l in range(i):
                    zi += dt * a[i][l] * k[l]
                k[i] = f(t[j] + dt * c[i],y[j] + zi)
                dy += dt * b[i] * k[i]
            y.append(y[j] + dy)
        return (t,y)


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
        y = np.zeros(n) ; y[0] = y0
        # Количество стадий #
        s = len(b)
        # Итерация по временной сетке (j) #
        for j in range(n-1):
            # Смещение стадийной касательной z #
            # Система нелинейных уравнений F(z)=0 #
            def F(z):
                F = [z[i] for i in range(len(z))]
                # Итерация по стадиям (i) #
                for i in range(len(z)):
                    # Итерация по всем (т.к. неявный метод) стадиям (l) #
                    for l in range(len(z)):
                        F[i] -= dt * a[i][l] * f(t[j] + dt * c[l],y[j] + z[l])
                return np.array(F)
            # Якобиан F(z) #
            def J(z):
                J = np.identity(len(z))
                # Итерация по стадиям (i) #
                for i in range(len(z)):
                    # Итерация по всем (т.к. неявный метод) стадиям (l) #
                    for l in range(len(z)):
                        J[i][l] -= dt * a[i][l] * df(t[j] + dt * c[l],y[j] + z[l])
                return np.array(J)
            # Начальное приближение z0 #
            z0 = np.zeros(s)
            # Метод Ньютона #
            z = newton(F, J, z0)
            # Смещение приближенного значения #
            dy = 0
            # Итерация по стадиям (i) #
            for i in range(s):
                dy += dt * b[i] * f(t[j] + dt * c[i],y[j] + z[i])
            y[j+1] = y[j] + dy
        return (t,y)