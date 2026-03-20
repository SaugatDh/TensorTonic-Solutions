import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    v = np.asarray(v)
    # n = v.shape[0]
    # Z = np.zeros((n,n))
    # for i in range(n):
    #     Z[i,i] = v[i]
    # return Z
    return np.diag(v)