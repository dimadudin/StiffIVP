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
            k = np.array([np.zeros_like(y0, dtype=np.float64) for _ in range(s)], dtype=np.float64)
            # Смещение приближенного значения #
            dy = np.zeros_like(y0, dtype=np.float64)
            # Итерация по стадиям (i) #
            for i in range(s):
                zi = np.zeros_like(y0, dtype=np.float64)
                # Итерация по предыдущим (т.к. явный метод) стадиям (l) #
                for l in range(i):
                    zi += dt * a[i,l] * k[l]
                k[i] = f(t[j] + dt * c[i],y[j] + zi)
                dy += dt * b[i] * k[i]
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