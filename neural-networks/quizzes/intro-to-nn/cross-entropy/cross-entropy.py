import numpy as np


def cross_entropy(y, p):
    y = np.float_(y)
    p = np.float_(p)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
