import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A = np.asarray(A,dtype=float)
    if A.ndim !=2:
        return None
    m,n=A.shape
    if m!=n:
        return None
    if np.linalg.det(A) == 0:
        return
    return np.linalg.inv(A)
