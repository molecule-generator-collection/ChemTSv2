import numpy as np


def minmax(x, min, max):
    return (x - min) / (max - min)


def max_gauss(x, a=1, mu=8, sigma=2):
    if x > mu:
        return 1
    else:
        return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def min_gauss(x, a=1, mu=2, sigma=2):
    if x < mu:
        return 1
    else:
        return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def rectangular(x, min, max):
    if min <= x <= max:
        return 1
    else:
        return 0
