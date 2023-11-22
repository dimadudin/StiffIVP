import numpy as np
from rungekutta import RK

METHODS = {'RK': RK}

def solveIVP(fun, t_span, y0, method='RK'):
    t0, tn = map(float, t_span)
    method = METHODS[method]
    solver = method(fun, t0, y0, tn)

    ts = [t0]
    ys = [y0]
    status = 'run'
    while status == 'run':
        status = solver.step()
        if status == 'fail':
            break
        ts.append(solver.t)
        ys.append(solver.y)
    ts = np.array(ts)
    ys = np.vstack(ys).T

    return dict(t=ts, y=ys, status=status)
