import random


def calc_error(thetas, ps, ys):
    # yhats = [0] * len(ps)
    # for c, xs in enumerate(ps):
    #     for t, x in zip(thetas, xs):
    #         yhats[c] += t * x

    yhats = [sum(t * x for t, x in zip(thetas, xs)) for xs in ps]
    error = 0
    for yh, y in zip(yhats, ys):
        error += (yh - y) ** 2
    return error / (2 * len(ys)), yhats

# def update_theta(thetas, ps, ys, yhats, alpha):
#     n = len(ys)
#     ds = [(yh - y) for yh, y in zip(yhats, ys)]
#     js = [0] * len(ys)
#     for c, xs in enumerate(ps):
#         for x, d in zip(xs, ds):
#             js[c] += (x * d)
#
#     js = [(alpha * i/n) for i in js]
#     return [t - j for t,j in zip(thetas, js)]

def update_theta(thetas, ps, ys, yhats, alpha):
    n = len(ys)
    ds = [(yh - y) for yh, y in zip(yhats, ys)]
    js = [sum(x * d for x, d in zip(xs, ds)) for xs in ps]
    return [t - alpha * j / n for t, j in zip(thetas, js)]


def linreg(ps: list[list[float]], ys: list[float], epsilon: float, alpha: float):
    # num_thetas = len(ps[0]) +1 # number of features

    for i in ps:
        i.insert(0, 1.0)

    # thetas = [random.uniform(-1, 1) for i in range(num_thetas)]
    thetas = [1.3, 2.9]

    prev_error = 0.0
    cur_error, yhats = calc_error(thetas, ps, ys)
    
    while abs(cur_error - prev_error) >= epsilon:
        thetas = update_theta(thetas, ps, ys, yhats, alpha)
        prev_error = cur_error
        cur_error, yhats = calc_error(thetas, ps, ys)

    return thetas

def calc_log_error(thetas, ps, ys):
    # yhats = [0] * len(ps)
    # for c, xs in enumerate(ps):
    #     for t, x in zip(thetas, xs):
    #         yhats[c] += t * x

    yhats = [sum(t * x for t, x in zip(thetas, xs)) for xs in ps]
    error = 0
    for yh, y in zip(yhats, ys):
        error += (yh - y) ** 2
    return error / (2 * len(ys)), yhats
def update_log_theta(thetas, ps, ys, yhats, alpha):
    n = len(ys)
    ds = [(yh - y) for yh, y in zip(yhats, ys)]
    js = [sum(x * d for x, d in zip(xs, ds)) for xs in ps]
    return [t - alpha * j / n for t, j in zip(thetas, js)]

def logreg(ps: list[list[float]], ys: list[float], epsilon: float, alpha: float):
    for i in ps:
        i.insert(0, 1.0)

    thetas = [0.8, 1.1]

    prev_error = 0.0
    cur_error, yhats = calc_log_error(thetas, ps, ys)
    
    while abs(cur_error - prev_error) >= epsilon:
        thetas = update_log_theta(thetas, ps, ys, yhats, alpha)
        prev_error = cur_error
        cur_error, yhats = calc_error(thetas, ps, ys)

    return thetas



ps = [[1.0], [3.0]]
ys = [2.0, 1.0]
thetas = linreg(ps, ys, 1*10**-7, 0.1)
print(thetas)
