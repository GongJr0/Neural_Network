import numpy as np

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x: float) -> float:
    return max(0, x)