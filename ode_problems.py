from numerical_ode import rungekutta
import numpy as np
import time
# Решение тестового уравнения явным методом Рунге-Кутта #
def solve_ode(problem= 'test', method='rkm3', tol=1e-3, dtmin=5.e-6, dtmax=10.0):
    # Параметры метода #
    mp = butcher_tables[method]
    rk = rungekutta(mp['a'], mp['b'], mp['c'], mp['p'])
    # Отображение графика оператора перехода и области стабильности метода #
    rk.plot_R()
    rk.plot_P()
    # Постановка задачи #
    pp = problems[problem]
    rk.init_problem(pp['f'], pp['y0'], pp['t0'], pp['tn'], pp['df'])
    # Параметры итерации #
    rk.init_iter(tol, dtmin, dtmax)
    st = time.time()
    # Решение заданной задачи инициализированным методом #
    t, y = rk()
    et = time.time()
    print(f'Время выполнения алгоритма = {et-st} секунд')
    # Отображение решения #
    rk.plot_Y(t, y)
    # Отображение изменения временных шагов #
    rk.plot_T(t)

# Параметры задач #
def f_test(t,y):
      return np.exp(y)*(1+t)
def df_test(t,y):
      return np.exp(y)*(1+t)
def f_robertson(t,y):
        return np.array([-0.04*y[0] + 1e4*y[1]*y[2],
                       0.04*y[0] - 1e4*y[1]*y[2] - 3e7*y[1]*y[1],
                                                   3e7*y[1]*y[1]], dtype=np.float64)
def df_robertson(t,y):
    return np.array([-0.04 + 1e4*y[2] +            1e4*y[1] +
                        0.04 - 1e4*y[2] - 6e7*y[1] - 1e4*y[1] +
                            0 +            6e7*y[1] +        0], dtype=np.float64)
def f_brunner(t,y):
        f = np.array([-0.013*y[1] - 1e3*y[0]*y[1] - 2.5e3*y[0]*y[2],
                      -0.013*y[1] - 1e3*y[0]*y[1]                  ,
                                                  - 2.5e3*y[0]*y[2]], dtype=np.float64)
        return f
def df_brunner(t,y):
        return np.array([-1e3*y[1] - 2.5e3*y[2] - 0.013 - 1e3*y[0] - 2.5e3*y[0] -
                          1e3*y[1]                      - 1e3*y[0]              -
                                     2.5e3*y[2]                    - 2.5e3*y[0]], dtype=np.float64)
problems = {
    "test": 
    {
        "f": f_test,
        "df": df_test,
        "t0": 0,
        "tn": 1,
        "y0": np.array([-0.5], dtype=np.float64),
    },
    "robertson": 
    {
        "f": f_robertson,
        "df": df_robertson,
        "t0": 0,
        "tn": 40,
        "y0": np.array([1,0,0], dtype=np.float64),
    },
    "brunner": 
    {
        "f": f_brunner,
        "df": df_brunner,
        "t0": 0,
        "tn": 50,
        "y0": np.array([0,1,1], dtype=np.float64),
    },
}

# Параметры используемых методов #
butcher_tables = {
    "rkm3": 
    {
        "a": np.array([[0,0,0],[1/2,0,0],[-1,2,0]], dtype=np.float64),
        "b": np.array([1/6,2/3,1/6], dtype=np.float64),
        "c": np.array([0,1/2,1], dtype=np.float64),
        "p": 3.,
    },
    "radau2a": 
    {
        "a": np.array([[5/12,-1/12],[3/4,1/4]], dtype=np.float64),
        "b": np.array([3/4,1/4], dtype=np.float64),
        "c": np.array([1/3,1], dtype=np.float64),
        "p": 3.,
    }
}
