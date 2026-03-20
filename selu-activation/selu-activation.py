import numpy as np

def selu(x, lam=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    """
    Apply SELU activation element-wise.
    Returns a list of floats rounded to 4 decimal places.
    """
    # Write code here
    result = []
    for val in x:
        if val > 0:
            result.append(float(val*lam))
        else:
            res = lam*alpha * (math.exp(val) - 1)
            result.append(float(res))
            
    return result
