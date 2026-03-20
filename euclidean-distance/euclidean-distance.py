import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x = np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)
    if x.shape != y.shape:
        raise ValueError("Vectors must have the same shape")
    return float(np.sqrt(np.sum((x-y)**2)))