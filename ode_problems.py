from numerical_ode import rungekutta
import numpy as np

def rk3_test(method):
    # # Решение тестового уравнения явным методом Рунге-Кутта #
    # Явный метод Рунге-Кутта 3-го порядка с адаптивным временным шагом #
    a = np.array([[0,0,0],[1/2,0,0],[-1,2,0]], dtype=np.float64)
    b = np.array([1/6,2/3,1/6], dtype=np.float64)
    c = np.array([0,1/2,1], dtype=np.float64)
    p = 3.
    ex_rk3 = rungekutta(a, b, c, p)
    # Отображение графика оператора перехода и области стабильности метода #
    ex_rk3.plot_R()
    ex_rk3.plot_P()
    # Постановка задачи #
    t0, tn = 0, 1
    y0 = np.array([-0.5], dtype=np.float64)
    f = lambda t,y: np.exp(y)*(1+t)
    ex_rk3.init_problem(f, y0, t0, tn)
    # Решение заданной задачи инициализированным методом #
    t, y = ex_rk3(0.01, method)
    # Отображение решения #
    ex_rk3.plot_Y(t, y)
    # Отображение изменения временных шагов #
    ex_rk3.plot_T(t)
    # Мета данные процесса приближения #
def robertson():
    # Решение задачи Робертсона Неявным методом Рунге-Кутта #
    # метод Радо IIA с адаптивным временным шагом #
    a = np.array([[5/12,-1/12],[3/4,1/4]], dtype=np.float64)
    b = np.array([3/4,1/4], dtype=np.float64)
    c = np.array([1/3,1], dtype=np.float64)
    p = 3.
    im_rk3 = rungekutta(a, b, c, p)
    # Отображение графика оператора перехода и области стабильности метода #
    im_rk3.plot_R()
    im_rk3.plot_P()
    # Параметры задачи #
    t0, tn = 0, 40
    y0 = np.array([1,0,0], dtype=np.float64)
    def f(t,y):
        f = np.array([-0.04*y[0] + 1e4*y[1]*y[2],
                       0.04*y[0] - 1e4*y[1]*y[2] - 3e7*y[1]*y[1],
                                                   3e7*y[1]*y[1]], dtype=np.float64)
        return f
    def df(t,y):
        return np.array([-0.04 + 1e4*y[2] +            1e4*y[1] +
                          0.04 - 1e4*y[2] - 6e7*y[1] - 1e4*y[1] +
                             0 +            6e7*y[1] +        0], dtype=np.float64)
    im_rk3.init_problem(f, y0, t0, tn, df)
    # Решение заданной задачи инициализированным методом #
    t, y = im_rk3(0.0005, "implicit")
    # Отображение решения #
    im_rk3.plot_Y(t, y)
    # Отображение изменения временных шагов #

    # Мета данные процесса приближения #
