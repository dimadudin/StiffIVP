# Реализация и отображение решения данных задач #
import numpy as np
import matplotlib.pyplot as plt
from numerical_ode import explicit_rk, implicit_rk, plot_R

if __name__ == "__main__":
    # # Решение тестового уравнения явным методом Рунге-Кутта #
    # # Явный метод Рунге-Кутта 3-го порядка с адаптивным временным шагом #
    # a = np.array([[0,0,0],[1/2,0,0],[-1,2,0]], dtype=np.float64)
    # b = np.array([1/6,2/3,1/6], dtype=np.float64)
    # c = np.array([0,1/2,1], dtype=np.float64)
    # ex_rk3 = explicit_rk(a, b, c)
    # # Отображение графика оператора перехода и области стабильности метода #
    # plot_R(a, b)
    # # Параметры задачи #
    # t0, tn = 0, 1
    # y0 = np.array([-0.5], dtype=np.float64)
    # f = lambda t,y: np.exp(y)*(1+t)
    # ex_rk3.init_problem(f, y0, t0, tn)
    # y_ex = lambda t: -np.log(np.exp(0.5) - t - t**2/2)
    # # Решение заданной задачи инициализированным методом #
    # t, y = ex_rk3(0.01)
    # # Отображение решения #
    # plt.figure(layout="constrained")
    # plt.plot(t, y, "r-", marker='o', lw=6, alpha=0.6,label="$y$", markersize="4")
    # plt.plot(t, y_ex(t), "b-", marker='.', lw=2, alpha=1,label="$y_t$", markersize="4")
    # plt.grid(True)
    # plt.xlabel("$t$")
    # plt.ylabel("$y$")
    # plt.legend(loc='best', frameon=False)
    # plt.show()
    # # Отображение изменения временных шагов #

    # # Мета данные процесса приближения #

    # Решение тестового уравнения неявным методом Рунге-Кутта #
    # метод Радо IIA с адаптивным временным шагом #
    a = np.array([[5/12,-1/12],[3/4,1/4]], dtype=np.float64)
    b = np.array([3/4,1/4], dtype=np.float64)
    c = np.array([1/3,1], dtype=np.float64)
    im_rk3 = implicit_rk(a, b, c)
    # Отображение графика оператора перехода и области стабильности метода #
    # plot_R(a, b)
    # Параметры задачи #
    t0, tn = 0, 1
    y0 = np.array([-0.5], dtype=np.float64)
    f = lambda t,y: np.array(np.exp(y)*(1+t), dtype=np.float64)
    im_rk3.init_problem(f, y0, t0, tn)
    df = lambda t,y: np.array([[np.exp(y)*(1+t)]], dtype=np.float64)
    y_ex = lambda t: -np.log(np.exp(0.5) - t - t**2/2)
    # Решение заданной задачи инициализированным методом #
    t, y = im_rk3(df, 0.01)
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

    # # Решение задачи Робертсона Неявным методом Рунге-Кутта #
    # # метод Радо IIA с адаптивным временным шагом #
    # a = np.array([[5/12,-1/12],[3/4,1/4]], dtype=np.float64)
    # b = np.array([3/4,1/4], dtype=np.float64)
    # c = np.array([1/3,1], dtype=np.float64)
    # im_rk3 = implicit_rk(a, b, c)
    # # Отображение графика оператора перехода и области стабильности метода #
    # plot_R(a, b)
    # # Параметры задачи #
    # t0, T = 0, 40
    # y0 = np.array([1,0,0], dtype=np.float64)
    # def f(t,y):
    #     f = np.array([-0.04*y[0] + 1e4*y[1]*y[2],
    #                       0.04*y[0] - 1e4*y[1]*y[2] - 3e7*y[1]**2,
    #                                                   3e7*y[1]**2], dtype=np.float64)
    #     return f
    # def df(t,y):
    #     return np.array([-0.04 + 1e4*y[2] +            1e4*y[1] +
    #                       0.04 - 1e4*y[2] - 6e7*y[1] - 1e4*y[1] +
    #                          0 +            6e7*y[1] +        0], dtype=np.float64)
    # # y_ex = lambda t: -np.log(np.exp(0.5) - t - t**2/2)
    # n = 1000
    # # Решение заданной задачи инициализированным методом #
    # t, y = im_rk3(f, df, y0, t0, T, n)
    # # Отображение решения #
    # plt.figure(layout="constrained")
    # plt.plot(t, y[:,0], "r-", marker='.', lw=2, alpha=0.6,label="$y_1$", markersize="4")
    # plt.plot(t, y[:,1], "b-", marker='.', lw=2, alpha=0.6,label="$y_2$", markersize="4")
    # plt.plot(t, y[:,2], "g-", marker='.', lw=2, alpha=0.6, label="$y_3$", markersize="4")
    # # plt.plot(t, y_ex(t), "b-", marker='.', lw=2, alpha=1,label="$y_t$", markersize="4")
    # plt.grid(True)
    # plt.xlabel("$t$")
    # plt.ylabel("$y$")
    # plt.legend(loc='best', frameon=False)
    # plt.show()
    # # Отображение изменения временных шагов #

    # # Мета данные процесса приближения #
