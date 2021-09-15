import numpy as np
from scipy.stats import norm


def breaking(p, m):
    p = p - 1
    # seleect p points out of the m points
    # then minimize the piecewise linear approximation L(*)

    u = np.linspace(start=-3, stop=3, endpoint=True, num=m)
    cache_cdf = list(map(lambda x: norm.cdf(x), u))

    cost = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            s = (cache_cdf[j] - cache_cdf[i]) / (u[j] - u[i])
            x_star = np.sqrt(-np.log(2 * np.pi) - 2 * np.log(s))
            x_star = x_star if u[i] <= x_star <= u[j] else -x_star
            if u[i] <= -x_star <= u[j]:
                cost1 = abs(norm.cdf(x_star) - (cache_cdf[i] + s * (x_star - u[i])))
                cost2 = abs(norm.cdf(-x_star) - (cache_cdf[i] + s * (-x_star - u[i])))
                cost[i, j] = max(cost1, cost2)
            else:
                cost[i, j] = abs(norm.cdf(x_star) - (cache_cdf[i] + s * (x_star - u[i])))

    D = np.zeros((p + 1, m))
    D[1, p-1:m-1] = cost[p-1:m-1, -1]
    for k in range(2, p + 1):
        for i in range(p - k, m - k):
            min_cost = np.inf
            for j in range(i + 1, m - k + 1):
                temp = D[k - 1, j] + cost[i, j]
                if temp < min_cost:
                    min_cost = temp
            D[k, i] = min_cost
    return D, cost


def backtracking(D, cost):
    p, m = D.shape
    p = p - 1
    u = np.linspace(start=-3, stop=3, endpoint=True, num=m)
    # backtracking
    optim_points = [-1000, ]
    optim_points.append(u[0])

    min_cost = D[p, 0]
    j = 0
    for k in range(p - 1, 0, -1):
        ind = np.argwhere(abs(min_cost - D[k, j + 1: m - k] - cost[j, j + 1:m - k]) < 1e-10)[0][0]
        j = j + ind + 1
        min_cost = D[k, j]
        optim_points.append(u[j])
    optim_points.append(u[-1])
    optim_points.append(1000)
    return optim_points


def find_break_points(p=6, m=100):
    D, cost = breaking(p=p, m=m)
    # print("the minimum loss of PWL is: ", D[-1, 0])
    v = backtracking(D, cost)
    return norm.cdf(v), v, D[-1, 0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    p = 6
    m = 100

    optim_points = find_break_points(p=p, m=m)
    print(optim_points, len(optim_points))


    u = np.linspace(start=-4, stop=4, endpoint=True, num=2*m)
    plt.plot(u, norm.cdf(u))
    for i in range(len(optim_points)-1):
        plt.plot(optim_points[i:i+2], norm.cdf(optim_points[i:i+2]))
    plt.show()



