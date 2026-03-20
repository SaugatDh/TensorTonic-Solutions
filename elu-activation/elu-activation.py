def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    result = []
    for val in x:
        if val >0:
            result.append(float(val))
        else:
            res = alpha * (math.exp(val)-1)
            result.append(float(res))

    return result
    