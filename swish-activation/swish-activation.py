import numpy as np

def swish(x):
    x = np.asarray(x, dtype=float)
    # Formula: x * sigmoid(x)
    return x * (1 / (1 + np.exp(-x)))