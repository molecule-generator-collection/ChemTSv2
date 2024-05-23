import numpy as np

def gauss(x, a=1, mu=8, sigma=2):
    return a * np.exp(-(x-mu)**2 / (2*sigma**2))


def minmax(x, min, max):
    return (x-min) / (max-min)


def max_gauss(x, a=1, mu=8, sigma=2):
    if x > mu:
        return 1
    else :
        return a * np.exp(-(x-mu)**2 / (2*sigma**2))


def min_gauss(x, a=1, mu=2, sigma=2):
    if x < mu:
        return 1
    else :
        return a * np.exp(-(x-mu)**2 / (2*sigma**2))


def rectangular(x, min, max):
    if min <= x <= max:
        return 1
    else:
        return 0

def trapezoid(x, top_min=200, top_max=300, bottom_min=100, bottom_max=400):
    if not (bottom_min <= top_min and top_min < top_max and top_max <= bottom_max):
        raise ValueError('check the input values.')
    if x < bottom_min:
        return 0
    elif bottom_min <= x and x < top_min:
        return (1/(top_min - bottom_min)) * (x-bottom_min)
    elif top_min <= x and x < top_max:
        return 1
    elif top_max <= x and x < bottom_max:
        return (1/(top_max - bottom_max)) * (x-bottom_max)
    elif bottom_max <= x:
        return 0

# def trapezoid(x, a=1, mu=8, sigma=2):
#     if x < (mu - 1.5*sigma):
#         return 0
#     elif (mu - 1.5*sigma) <= x and x < (mu - 1.0*sigma):
#         return (2 * x / sigma ) - (2 * mu / sigma) +3.0
#     elif (mu - 1.0*sigma) <= x and x < (mu + 1.0*sigma):
#         return 1
#     elif (mu + 1.0*sigma) <= x and x < (mu + 1.5*sigma):
#         return (-2 * x / sigma ) + (2 * mu / sigma) +3.0
#     elif (mu + 1.5*sigma) <= x:
#         return 0
    