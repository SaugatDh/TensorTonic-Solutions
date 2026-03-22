import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a matrix using basic NumPy operations with error handling.
    """
    try:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2:
            return None
        # Validate axis bounds to satisfy test requirements
        if axis is not None:
            if axis >= matrix.ndim or axis < -matrix.ndim:
                return None

        # 1. Calculate the norm using NumPy primitives
        if norm_type == 'l2':
            # Euclidean Norm: sqrt(sum(x^2))
            norm = np.sqrt(np.sum(matrix**2, axis=axis, keepdims=True))
        elif norm_type == 'l1':
            # Manhattan Norm: sum(|x|)
            norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        elif norm_type == 'max':
            # Infinity Norm: max(|x|)
            norm = np.max(np.abs(matrix), axis=axis, keepdims=True)
        else:
            raise ValueError("Unsupported norm_type.")

        # 2. Prevent division by zero
        norm = np.where(norm == 0, 1, norm)

        # 3. Broadcasting division
        return matrix / norm

    except (ValueError, TypeError, ZeroDivisionError):
        # Return None if input is invalid or a mathematical error occurs
        return None