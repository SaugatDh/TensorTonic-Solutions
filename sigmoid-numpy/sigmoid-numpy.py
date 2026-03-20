import numpy as np

def sigmoid(x):
    # Standard formula: 1 / (1 + e^-x)
    x = np.asarray(x)
    return 1 / (1 + np.exp(-x))

x = np.array([1, 2, 3])
y = sigmoid(x)
print(y)