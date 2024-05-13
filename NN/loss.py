import numpy as np

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)

def rmse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mse_loss(y_true, y_pred))

def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    def softmax(x: np.ndarray) -> np.ndarray:
        exp = np.exp(x)
        return exp / np.sum(exp)
    
    y_pred = softmax(y_pred)
    loss: float;
    
    for i in range(len(y_pred)):
        loss = loss + (-1*y_true[i]*np.log(y_pred[i]))
    return loss