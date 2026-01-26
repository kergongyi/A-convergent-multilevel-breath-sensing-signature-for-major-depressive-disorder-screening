import numpy as np


def Cliffs_delta(x, y):
    x = np.array(x)
    y = np.array(y)
    nx, ny = len(x), len(y)

    diff = np.subtract.outer(x, y)
    n_greater = np.sum(diff > 0)
    n_less = np.sum(diff < 0)
    delta = (n_greater - n_less) / (nx * ny)

    return delta
