import numpy as np


def softmax(values):
    exp_l = np.exp(values)
    return np.divide(exp_l, exp_l.sum())
