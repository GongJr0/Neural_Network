import numpy as np
import NN.activation as act
import NN.loss as loss

from typing import Callable, Sequence
from typing import List

class Neuron:
    def __init__(self, weights: Sequence[float | int], bias: float, activation: Callable[[float], float] = act.sigmoid) -> None:
        self.weights = weights
        self.bias = bias

        self.activate = activation
    def forward(self, inputs: Sequence[float | int]) -> float:
        out = np.dot(inputs, self.weights) + self.bias
        return self.activate(out)
    
class NNetwork:
    def __init__(self, activate: Callable[[float], float] = act.sigmoid) -> None:
        self.activate = activate
        
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        
    def forward(self, input: np.ndarray) -> float:
        hidden_1 = self.activate(self.w1 * input[0] + self.w2 * input[1] + self.b1)
        hidden_2 = self.activate(self.w3 * input[0] + self.w4 * input[1] + self.b2)
        
        output = self.activate(self.w5 * hidden_1 + self.w6 * hidden_2 + self.b3)
        return output
    def train(self, preds: np.ndarray, trues: np.ndarray, lr: float = 0.1, epochs: int = 1000) -> None:
        for epoch in range(epochs):
            for pred, true in zip(preds, trues):
                sum_hidden_1 = self.w1 * pred[0] + self.w2 * pred[1] + self.b1
                hidden_1 = self.activate(sum_hidden_1)
                
                sum_hidden_2 = self.w3 * pred[0] + self.w4 * pred[1] + self.b2
                hidden_2 = self.activate(sum_hidden_2)
                
                sum_output = self.w5 * hidden_1 + self.w6 * hidden_2 + self.b3
                output = self.activate(sum_output)
                
                y_pred = output
                
                #partial derivatives
                d_L_d_ypred = -2 * (true - pred)
                
                #o1
                d_ypred_d_w5 = hidden_1 * act.derivative_sigmoid(sum_output)
                d_ypred_d_w6 = hidden_2 * act.derivative_sigmoid(sum_output)
                
                d_ypred_d_b3 = act.derivative_sigmoid(sum_output)
                
                d_ypred_d_h1 = self.w5 * act.derivative_sigmoid(sum_output)
                d_ypred_d_h2 = self.w6 * act.derivative_sigmoid(sum_output)
                
                #h1
                d_h1_d_w1 = pred[0] * act.derivative_sigmoid(sum_hidden_1)
                d_h1_d_w2 = pred[1] * act.derivative_sigmoid(sum_hidden_1)
                d_h1_d_b1 = act.derivative_sigmoid(sum_hidden_1)
                
                #h2
                d_h2_d_w3 = pred[0] * act.derivative_sigmoid(sum_hidden_2)
                d_h2_d_w4 = pred[1] * act.derivative_sigmoid(sum_hidden_2)
                d_h2_d_b2 = act.derivative_sigmoid(sum_hidden_2)
                
                
                #Weight updates
                
                #h1
                self.w1 -= lr * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= lr * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= lr * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                
                #h2
                self.w3 -= lr * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= lr * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= lr * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                
                #o1
                self.w5 -= lr * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= lr * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= lr * d_L_d_ypred * d_ypred_d_b3
                
                #loss
                if epoch % 10 == 0:
                    new_pred: np.ndarray = np.apply_along_axis(self.forward, 1, pred)
                    loss = loss.mse_loss(true, new_pred)
                    print(f'Epoch {epoch}, Loss (MSE): {loss}')
                                
            