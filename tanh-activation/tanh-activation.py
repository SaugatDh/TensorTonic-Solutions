import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x=np.asarray(x,dtype=float)
    exp_p = np.exp(x)
    exp_n= np.exp(-x)
    res = (exp_p-exp_n)/(exp_p+exp_n)

    if x.ndim == 0:
        return np.array([res])

    return res
x=5
res = tanh(x)
print(res)