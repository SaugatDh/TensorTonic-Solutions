import numpy as np
import math

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (1, seq_len, d_model) using sin/cos formulation.
    Even indices -> sin, odd indices -> cos.
    """
    pe = np.zeros((seq_len, d_model)) 
    
    position = np.arange(seq_len, dtype=float).reshape(seq_len, 1)
    
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(base) / d_model)).reshape(1, -1)
    
    pe[:, 0::2] = np.sin(position * div_term)
    
    pe[:, 1::2] = np.cos(position * div_term[:, :d_model//2])
    
    pe = pe.reshape(seq_len, d_model)
    
    return pe

