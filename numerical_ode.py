# Реализация методов Рунге-Кутты для решения ОДУ #
import numpy as np
from numerical_nonlinear import newton
from numpy.linalg import det, norm
import matplotlib.pyplot as plt

# Явный метод Рунге-Кутты #
class rungekutta:
    # Инициальизация метода #
    def __init__(self, a, b, c, p):
        # Параметры метода #
        self.a, self.b, self.c, self.s, self.p = a, b, c, len(b), p
    # Инициализация задачи #
    def init_problem(self, f, y0, t0, tn, df=None):
        # Параметры задачи #
        self.f, self.y0, self.t0, self.tn, self.df = f, y0, t0, tn, df
    # Итериция явного метода #
    def explicit(self, tj, yj, dt):
        a, b, c, s = self.a, self.b, self.c, self.s
        f = self.f
        k = np.array([np.zeros_like(yj, dtype=np.float64) for _ in range(s)], dtype=np.float64)
        dy = np.zeros_like(yj, dtype=np.float64)
        for i in range(s):
            zi = np.zeros_like(yj, dtype=np.float64)
            for l in range(i):
                zi += dt * a[i,l] * k[l]
            k[i] = f(tj + dt * c[i],yj + zi)
            dy += dt * b[i] * k[i]
        return dy
    # Итериция неявного метода #
    def implicit(self, tj, yj, dt):
        a, b, c, s = self.a, self.b, self.c, self.s
        f, df = self.f, self.df
        def F(z):
                f_ = np.array([z[i] for i in range(s)], dtype=np.float64)
                for i in range(s):
                    for l in range(s):
                        f_[i] -= dt * a[i,l] * f(tj + dt * c[l],yj + z[l])
                return f_
        def J(z):
            j_ = np.identity(s, dtype=np.float64)
            for i in range(s):
                for l in range(s):
                    j_[i][l] -= dt * a[i,l] * df(tj + dt * c[i],yj + z[l])
            return j_
        z0 =  np.array([np.zeros_like(yj) for _ in range(s)], dtype=np.float64)
        z = newton(F, J, z0)
        dy = np.zeros_like(yj, dtype=np.float64)
        for i in range(s):
            dy += dt * b[i] * f(tj + dt * c[i],yj + z[i])
        return dy
    # Решение задачи #
    def __call__(self, dt, iterm="explicit", tol=1e-5):
        y0, t0, tn = self.y0, self.t0, self.tn
        if iterm == "explicit": method = self.explicit
        elif iterm == "implicit": method = self.implicit
        else: return ("kys","fag")
        t, y = [t0], [y0]
        while(t[-1] < tn):
            tj, yj = t[-1], y[-1]
            dy = method(tj, yj, dt)
            dt_ = 0.5 * dt
            dy_ = method(tj, yj, dt_)
            dy_ += method(tj + dt_, yj + dy_, dt_)
            err = norm(dy - dy_)
            if err <= tol:
                t.append(tj + dt)
                y.append(yj + dy_)
                dt = min(dt*min(5.0, max(5.e-4, 0.8*(tol/err)**(1/(self.p)))), abs(tn-t0))
            else: dt = dt_
        return (np.array(t, dtype=np.float64), np.array(y, dtype=np.float64))
    # Отображение области устойчивости метода #
    def plot_R(self, x=np.linspace(-3,3,121), y=np.linspace(-3,3,121)):
        a, b = self.a, self.b
        newparams = {'axes.grid': True,
                'lines.markersize': 8, 'lines.linewidth': 2,
                'font.size': 14}
        plt.rcParams.update(newparams)
        X,Y = np.meshgrid(x,y)
        r_ = np.identity(len(x), dtype=np.float64)
        for i in range(len(x)):
            for j in range(len(y)):
                E = np.identity(len(a), dtype=np.float64)
                EzA = E - (x[i]+y[j]*1j) * a
                delta = det(EzA)
                delta1 = det(EzA + (x[i]+y[j]*1j)
                                * np.outer(np.ones(len(b)), b))
                r_[j][i] = abs(delta1/delta)
        plt.figure()
        plt.contourf(X,Y,r_, 1, levels = [0,1])
        plt.contour(X,Y,r_, 1, colors = 'black', levels = [1])
        plt.plot([min(x),max(x)],[0,0], 'k--')
        plt.plot([0,0],[min(y),max(y)], 'k--')
        plt.xlabel('Re(z)')
        plt.ylabel('Im(z)')
        plt.show()
    # Отображение оператора перехода метода #
    def plot_P(self, x=np.linspace(-3,3,121)):
        a, b = self.a, self.b
        newparams = {'axes.grid': True,
                'lines.markersize': 8, 'lines.linewidth': 2,
                'font.size': 14}
        plt.rcParams.update(newparams)
        p_ = np.zeros(len(x), dtype=np.float64)
        for i in range(len(x)):
            E = np.identity(len(a), dtype=np.float64)
            EzA = E - x[i] * a
            delta = det(EzA)
            delta1 = det(EzA + x[i] * np.outer(np.ones(len(b)), b))
            p_[i] = delta1/delta
        plt.figure()
        plt.plot(x, p_)
        plt.xlabel('x')
        plt.ylabel('P(x)')
        plt.show()
    