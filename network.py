import cupy

import numpy as np
from tqdm import tqdm

from utils import Layer, FLOAT_TYPE
from activations import SoftMax
from layers import *
from transformer_adapters import *


class CrossEntropy:

    """Fused softmax + cross entropy.

    gradients() returns d(loss)/d(the softmax layer's input), which is why
    SoftMax(fused_loss = True) passes its gradient straight through.

    Neither method builds a one hot. The identity matrix this used to hold is
    num_classes squared - 256 GiB at Gemma's 262144 token vocabulary - and indexing
    it expanded to a full (batch, sequence, vocab) array on every call. Each label
    selects exactly one entry per row, so both operations are a gather instead.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.positions = {}

    def _index(self, labels):

        """Index tuple addressing the entry each label selects.

        Broadcastable ranges over the leading axes, then the labels themselves, so
        this works for (batch,) classification labels and (batch, sequence) token
        labels alike. Cached per shape; batches only vary on the final partial one.
        """

        shape = labels.shape

        if shape not in self.positions:
            self.positions[shape] = tuple(
                cupy.arange(size).reshape((-1,) + (1,) * (len(shape) - axis - 1))
                for axis, size in enumerate(shape))

        return self.positions[shape] + (labels,)

    def gradients(self, logits, labels):
        # softmax output minus the one hot, without the one hot. Every (row, label)
        # pair is distinct, so the subtraction needs no scattered accumulation.
        gradient = logits.copy()
        gradient[self._index(labels)] -= 1
        return gradient

    def loss(self, logits, labels):
        eta = 1e-7
        return -1 * cupy.sum(cupy.log(logits[self._index(labels)] + eta))


class Network:

    def __init__(self, layers:list[Layer]):
        self.layers = layers

        # a fused SoftMax returns its gradient unchanged, which is only right when
        # CrossEntropy produced that gradient for the last layer of the network
        for position, layer in enumerate(layers[:-1]):
            if isinstance(layer, SoftMax) and layer.fused_loss:
                raise ValueError(
                    f"SoftMax at position {position} of {len(layers)} has fused_loss = True but is "
                    "not the final layer. Its backward pass returns the gradient unchanged, which "
                    "is only correct when CrossEntropy supplied it. Use SoftMax(fused_loss = False) "
                    "to apply the real Jacobian mid network.")
        
    def predict(self, input):
        return self._forward(input)

    def _forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
            if layer.inference_only:
                # the next layer already holds what it needs, so drop this one's
                # activations now rather than pinning all of them until the forward ends
                layer.clear_cache()
        return input

    def _backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def set_eval(self, eval_mode):
        for layer in self.layers:
            layer.set_eval(eval_mode)
            layer.zero_grad()
            
    def _zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
    
    def _zero_adam(self):
        for layer in self.layers:
            layer.zero_adam()

    def _update(self, learning_rate, weight_decay, t, num_samples):

        beta1, beta2 = 0.9, 0.999

        for layer in self.layers:
            
            for param, grad, moment, variance in zip(layer.parameters,
                                                     layer.gradients,
                                                     layer.moments,
                                                     layer.variances):
                grad /= num_samples
                lmda = weight_decay
                
                if len(param.shape) == 1 or isinstance(layer, (VitProjector, VitMLPHead, 
                                                               GPTEmbedFront, GPTEmbedBack)):
                    lmda = 0.0
                    
                moment *= beta1
                moment += (1 - beta1)*grad

                # grad is scratch from here on: it is zeroed immediately after the
                # step, so squaring and scaling it in place saves two full sized
                # temporaries per parameter. Bitwise identical to (1 - beta2)*grad**2.
                grad *= grad
                grad *= (1 - beta2)

                variance *= beta2
                variance += grad

                mom_hat = moment / (1 - beta1**t)
                var_hat = variance / (1 - beta2**t)

                param -= learning_rate * (mom_hat / (var_hat**0.5 + 1e-7) + lmda * param)

    def train(self, criterion, train_data, train_labels, test_data = None, test_labels = None,
                    augments = None, epochs = 1, batch_size = 64, batches_per_step = 1,
                    learning_rate = 0.001, weight_decay = 0.01):

        if any(layer.inference_only for layer in self.layers):
            raise RuntimeError("this model was built inside inference_mode(): it has no gradient, "
                               "moment or variance buffers to train with. Rebuild it outside the "
                               "context to train.")

        step_count = 0
        samples_per_step = batch_size * batches_per_step
        
        self._zero_adam()
        
        for i in range(epochs):
            
            self.set_eval(False)

            batches_seen = 0
            train_loss, train_correct = 0, 0

            shuffle = np.random.permutation(len(train_labels))
            train_data   = train_data[shuffle]
            train_labels = train_labels[shuffle]

            for j in tqdm(range(0, len(train_data), batch_size)):
                
                x = train_data  [j: min(j + batch_size, len(train_data))]
                y = train_labels[j: min(j + batch_size, len(train_data))]
                
                if augments is not None:
                    x = augments(x)

                x = cupy.array(x)
                y = cupy.array(y)
                
                y_hat = self._forward(x)
                
                grad = criterion.gradients(y_hat, y)
                self._backward(grad)
                
                batches_seen += 1
                if batches_seen % batches_per_step == 0:
                    step_count += 1
                    self._update(learning_rate, weight_decay, step_count, samples_per_step)
                    self._zero_grad()

                train_loss += criterion.loss(y_hat, y)
                train_correct += cupy.equal(cupy.argmax(y_hat, axis = -1), y).astype(cupy.int32).sum()
                
            train_loss     = train_loss    / np.prod(train_labels.shape)
            train_accuracy = train_correct / np.prod(train_labels.shape)
            print("epoch:", i + 1, "train loss:", train_loss, "train accuracy:", train_accuracy)
            
            if test_data is not None and test_labels is not None:
                test_loss, test_accuracy = self.evaluate(test_data, test_labels, criterion, batch_size=batch_size)
                print("epoch:", i + 1, "test loss:", test_loss, "test accuracy:", test_accuracy, "\n")

    def evaluate(self, test_data, test_labels, criterion, batch_size = 64):
        
        self.set_eval(True)
        loss, correct = 0, 0

        for i in tqdm(range(0, len(test_data), batch_size)):

            x = cupy.array(test_data  [i: min(i + batch_size, len(test_data))])
            y = cupy.array(test_labels[i: min(i + batch_size, len(test_data))])

            y_hat = self.predict(x)

            loss += criterion.loss(y_hat, y)
            correct += cupy.equal(cupy.argmax(y_hat, axis = -1), y).astype(cupy.int32).sum()

        return loss / np.prod(test_labels.shape), correct / np.prod(test_labels.shape)