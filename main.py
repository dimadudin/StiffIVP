# Реализация и отображение решения данных задач #
import numpy as np
import matplotlib.pyplot as plt
from numerical_ode import explicit_rk, implicit_rk

if __name__ == "__main__":
    # Решение задачи Робертсона явным методом Рунге-Кутта #
    # Явный метод Рунге-Кутта 3-го порядка с адаптивным временным шагом #
    a = [[0,0,0],[1/2,0,0],[-1,2,0]]
    b = [1/6,2/3,1/6]
    c = [0,1/2,1]
    ex_rk3 = explicit_rk(a, b, c)
    # Параметры задачи #
    t0, T = 0, 1
    y0 = -0.5
    f = lambda t,y: np.exp(y)*(1+t)
    y_ex = lambda t: -np.log(np.exp(0.5) - t - t**2/2)
    n = 100
    # Решение заданной задачи инициализированным методом #
    t, y = ex_rk3(f, y0, t0, T, n)
    # Отображение решения #
    plt.figure(layout="constrained")
    plt.plot(t, y, "r-", marker='o', lw=6, alpha=0.6,label="$y$", markersize="4")
    plt.plot(t, y_ex(t), "b-", marker='.', lw=2, alpha=1,label="$y_t$", markersize="4")
    plt.grid(True)
    plt.xlabel("$t$")
    plt.ylabel("$y$")
    plt.legend(loc='best', frameon=False)
    plt.show()
    # Отображение изменения временных шагов #

    # Мета данные процесса приближения #

    # Решение задачи Робертсона Неявным методом Рунге-Кутта #
    # метод Радо IIA с адаптивным временным шагом #
    a = [[5/12,-1/12],[3/4,1/4]]
    b = [3/4,1/4]
    c = [1/3,1]
    im_rk3 = implicit_rk(a, b, c)
    # Параметры задачи #
    t0, T = 0, 1
    y0 = -0.5
    f = lambda t,y: np.exp(y)*(1+t)
    df = lambda t,y: np.exp(y)*(1+t)
    y_ex = lambda t: -np.log(np.exp(0.5) - t - t**2/2)
    n = 100
    # Решение заданной задачи инициализированным методом #
    t, y = im_rk3(f, df, y0, t0, T, n)
    # Отображение решения #
    plt.figure(layout="constrained")
    plt.plot(t, y, "r-", marker='o', lw=6, alpha=0.6,label="$y$", markersize="4")
    plt.plot(t, y_ex(t), "b-", marker='.', lw=2, alpha=1,label="$y_t$", markersize="4")
    plt.grid(True)
    plt.xlabel("$t$")
    plt.ylabel("$y$")
    plt.legend(loc='best', frameon=False)
    plt.show()
    # Отображение изменения временных шагов #

    # Мета данные процесса приближения #
