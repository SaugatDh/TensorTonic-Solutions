import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    x = np.asarray(x)
    y=np.asarray(y)
    if len(x)!= len(y):
        raise ValueError("Vectors must have same length")
    result = 0.0
    for i in range(len(x)):
        result += float(x[i])*float(y[i])
    return result