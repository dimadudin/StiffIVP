# Реализация методов Рунге-Кутты для решения ОДУ #
import numpy as np

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
        y = np.zeros_like(range(n)) ; y[0] = y0
        # Количество стадий #
        s = len(b)
        # Итерация по временной сетке (j) #
        for j in range(n-1):
            # Смещение стадийной касательной #
            z = np.zeros_like(range(s))
            # Итерация по стадиям (i) #
            for i in range(s):
                # Итерация по предыдущим (т.к. явный метод) стадиям (l) #
                for l in range(i):
                    z[i] += dt * a[i,l] * f(t[j] + dt * c[l],y[j] + z[l])
            # Смещение приближенного значения #
            dy = 0
            # Итерация по стадиям (i) #
            for i in range(s):
                dy += dt * b[i] * f(t[j] + dt * c[i],y[j] + z[i])
            y[j+1] = y[j] + dy
        return (t,y)


class implicit_rk:
    def __init__(self, a, b, c):
        # Таблица Бутчера #
        self.a, self.b, self.c = a, b, c
    def __call__(self, args):
        # Таблица Бутчера #
        a, b, c = self.a, self.b, self.c
        # TODO: Реализовать #