import cupy

import numpy as np
from tqdm import tqdm

from utils import Layer, FLOAT_TYPE
from layers import *
from transformer_adapters import *


class CrossEntropy:

    def __init__(self, num_classes = -1):
        self.one_hot = None
        if num_classes > 0:
            self.one_hot = cupy.eye(num_classes, dtype = FLOAT_TYPE)

    def gradients(self, output, labels):
        if self.one_hot is not None:
            return output - self.one_hot[labels]
        else:
            one_hot = cupy.eye(output.shape[-1], dtype = FLOAT_TYPE)
            return output - one_hot(labels)
      
    def loss(self, output, labels):
        if self.one_hot is not None:
            labels = self.one_hot[labels]
        else:
            one_hot = cupy.eye(output.shape[-1], dtype = FLOAT_TYPE)
            labels = one_hot[labels]
        return -1 * cupy.sum(labels * cupy.log(output + 1e-7))


class Network:

    def __init__(self, layers:list[Layer]):
        self.layers = layers
        
    def predict(self, input):
        return self._forward(input)

    def _forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def _backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def set_eval(self, eval_mode = True):
        for layer in self.layers:
            layer.zero_grad()
            layer.set_eval(eval_mode)

    def _zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def _update(self, learning_rate, weight_decay, t):

        beta1, beta2 = 0.9, 0.99

        for i, layer in enumerate(self.layers):
            
            for param, grad, moment, variance in zip(layer.parameters,
                                                     layer.gradients,
                                                     layer.moments,
                                                     layer.variances):
                lmda = weight_decay 
                if len(param.shape) == 1 or isinstance(layer, (VitProjector, VitMLPHead, 
                                                               GPTEmbedFront, GPTEmbedBack)):
                    lmda = 0.0
                    
                moment *= beta1
                moment += (1 - beta1)*grad

                variance *= beta2
                variance += (1 - beta2)*grad**2

                mom_hat = moment / (1 - beta1**t)
                var_hat = variance / (1 - beta2**t)

                param -= learning_rate * (mom_hat / (var_hat**0.5 + 1e-7) + lmda * param)

    def train(self, data, labels, criterion, epochs = 1, batch_size = 64, learning_rate = 0.001, weight_decay = 0.01):
    
        t = 1
        for i in range(epochs):

            loss, correct = 0, 0
            
            for j in tqdm(range(0, len(data), batch_size)):
                
                x = cupy.array(data  [j: min(j + batch_size, len(data))])
                y = cupy.array(labels[j: min(j + batch_size, len(data))])
                
                self._zero_grad()
                
                y_hat = self._forward(x)
                grad = criterion.gradients(y_hat, y)

                self._backward(grad)
                self._update(learning_rate, weight_decay, t,)
                t += 1

                loss += criterion.loss(y_hat, y)
                correct += cupy.equal(cupy.argmax(y_hat, axis = -1), y).astype(cupy.int32).sum()

            print("epoch:", i + 1, "loss:", loss / np.prod(labels.shape), "accuracy:", correct / np.prod(labels.shape), "\n")

    def evaluate(self, test_data, test_labels, criterion, batch_size = 64):

        loss, correct = 0, 0

        for i in tqdm(range(0, len(test_data), batch_size)):

            x = cupy.array(test_data  [i: min(i + batch_size, len(test_data))])
            y = cupy.array(test_labels[i: min(i + batch_size, len(test_data))])

            y_hat = self.predict(x)

            loss += criterion.loss(y_hat, y)
            correct += cupy.equal(cupy.argmax(y_hat, axis = -1), y).astype(cupy.int32).sum()

        return loss / np.prod(test_labels.shape), correct / np.prod(test_labels.shape)