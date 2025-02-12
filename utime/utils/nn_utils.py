"""Utility functions for neural networks"""

import numpy as np

def softmax(logits, axis=-1):
    """Compute the numerically stable softmax function"""
    logits_stable = logits - np.max(logits, axis=axis, keepdims=True)
    exp_values = np.exp(logits_stable)
    sum_exp = np.sum(exp_values, axis=axis, keepdims=True)
    probabilities = exp_values / sum_exp
    return probabilities