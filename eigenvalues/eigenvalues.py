import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        matrix = np.asarray(matrix, dtype=float)
    except (ValueError, TypeError):
        return None
    if matrix.ndim != 2:
        return None
    m, n = matrix.shape
    if m != n or m == 0:
        return None
    try:
        eigvals = np.linalg.eigvals(matrix)
        idx = np.lexsort((eigvals.imag, eigvals.real))
        return eigvals[idx]
    except np.linalg.LinAlgError:
        return None