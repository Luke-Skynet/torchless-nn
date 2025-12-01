import cupy

from utils import Layer, FLOAT_TYPE, init_random_tensor, init_zeros_tensor

class Convolution(Layer):

    def __init__(self, input_size: tuple, kernel_size: tuple, num_kernels, padding = (0, 0)):
        super(Convolution, self).__init__()

        self.kernel_size =  kernel_size
        self.output_dims = (input_size[0] - self.kernel_size[0] + 1 + 2*padding[0],
                            input_size[1] - self.kernel_size[1] + 1 + 2*padding[1])
        
        self.pad_h, self.pad_w = padding
        self.padded_input = None

        self.weights = init_random_tensor((kernel_size[0], kernel_size[1], input_size[2], num_kernels))
        self.weights = self.weights / (input_size[2] * kernel_size[0] * kernel_size[1])**0.5
        self.bias    = init_zeros_tensor(num_kernels)
        
        self.weight_grads   = init_zeros_tensor(self.weights.shape)
        self.weight_moments = init_zeros_tensor(self.weights.shape)
        self.weight_vars    = init_zeros_tensor(self.weights.shape)

        self.bias_grads   = init_zeros_tensor(self.bias.shape)
        self.bias_moments = init_zeros_tensor(self.bias.shape)
        self.bias_vars    = init_zeros_tensor(self.bias.shape)

        self.parameters = [self.weights,          self.bias]
        self.gradients  = [self.weight_grads,     self.bias_grads]
        self.moments    = [self.weight_moments,   self.bias_moments]
        self.variances  = [self.weight_vars,      self.bias_vars]

    def forward(self, input):

        self.input = input
        
        self.padded_input = cupy.pad(self.input, ((0, 0), (self.pad_h, self.pad_h), (self.pad_w, self.pad_w), (0, 0)))

        self.output = cupy.lib.stride_tricks.sliding_window_view(self.padded_input, self.kernel_size, (1, 2))
        self.output = cupy.tensordot(self.output, self.weights, axes=((4, 5, 3), (0, 1, 2)))
        self.output = self.output + self.bias

        return self.output

    def backward(self, gradient):

        convolve = cupy.lib.stride_tricks.sliding_window_view(self.padded_input, self.output_dims, (1, 2))
        self.weight_grads += cupy.tensordot(convolve, gradient, axes=((0, 4, 5), (0, 1, 2))) / gradient.shape[0]

        self.bias_grads += cupy.mean(gradient, axis=(0, 1, 2))

        grad_pad_h = self.kernel_size[0] - 1
        grad_pad_w = self.kernel_size[1] - 1

        flipped_weights = cupy.flip(self.weights, axis=(0,1))

        gradient = cupy.pad(gradient, ((0, 0), (grad_pad_h, grad_pad_h), (grad_pad_w, grad_pad_w), (0, 0)))
        gradient = cupy.lib.stride_tricks.sliding_window_view(gradient, self.kernel_size, (1, 2))
        gradient = cupy.tensordot(gradient, flipped_weights, axes=((4, 5, 3), (0, 1, 3)))
        gradient = gradient[:, self.pad_h:gradient.shape[1]-self.pad_h, self.pad_w:gradient.shape[2]-self.pad_w, :]
        
        return gradient


class BatchNorm(Layer):
    def __init__(self, num_channels):
        super(BatchNorm, self).__init__()
        
        self.channels = num_channels
        self.eps = 1e-5
        
        self.momentum = 0.1
        
        self.running_mean = init_zeros_tensor((1, 1, 1, num_channels))
        self.running_var  = init_zeros_tensor((1, 1, 1, num_channels))

        self.gamma = cupy.ones(num_channels, dtype = FLOAT_TYPE)
        self.beta  = init_zeros_tensor(num_channels)

        self.gamma_grads = init_zeros_tensor(num_channels)
        self.gamma_moments = init_zeros_tensor(num_channels)
        self.gamma_vars    = init_zeros_tensor(num_channels)

        self.beta_grads  = init_zeros_tensor(num_channels)
        self.beta_moments = init_zeros_tensor(num_channels)
        self.beta_vars    = init_zeros_tensor(num_channels)

        self.mean = None
        self.var = None
        self.std = None
        self.centered = None
        self.normed = None

        self.parameters = [self.gamma,         self.beta]
        self.gradients  = [self.gamma_grads,   self.beta_grads]
        self.moments    = [self.gamma_moments, self.beta_moments]
        self.variances  = [self.gamma_vars,    self.beta_vars]
    
    def forward(self, input):
        
        self.input = input
        
        if self.eval_mode is False:
            self.mean = cupy.mean(input, axis = (0, 1, 2), keepdims = True)
            self.var  = cupy.var(input, axis = (0, 1, 2), keepdims = True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * self.var
        else:
            self.mean = self.running_mean
            self.var = self.running_var
            
        self.centered = self.input - self.mean
        
        self.std = (self.var + self.eps)**0.5
        self.normed = self.centered / self.std
        
        self.output = self.gamma * self.normed + self.beta
        return self.output
            
    def backward(self, gradient):
        
        self.gamma_grads += cupy.sum(gradient * self.normed, axis=(0, 1, 2)) / gradient.shape[0]
        self.beta_grads  += cupy.sum(gradient, axis=(0, 1, 2)) / gradient.shape[0]
        
        gradient = gradient * self.gamma
        
        gradient_normed = (gradient - cupy.mean(gradient, axis = (0, 1, 2), keepdims = True)) / self.std
        
        return gradient_normed - self.centered * (cupy.mean(gradient * self.centered, axis = (0, 1, 2), keepdims = True) / self.std**3)
    

class MaxPool(Layer):

    def __init__(self, pool_height, pool_width):
        super(MaxPool, self).__init__()

        self.pool_h, self.pool_w = (pool_height, pool_width)
        self.batch_size, self.in_h, self.in_w, self.channels = 0,0,0,0
        self.mask = None

    def forward(self, input):

        self.input = input
        self.batch_size, self.in_h, self.in_w, self.channels = self.input.shape

        out_h = self.in_h // self.pool_h
        out_w = self.in_w // self.pool_w

        view = self.input[:, :out_h*self.pool_h, :out_w*self.pool_w, :]
        view = view.reshape(self.batch_size, out_h, self.pool_h, out_w, self.pool_w, self.channels)
        
        self.output = view.max(axis=(2, 4), keepdims = True)

        self.mask   = self.output == view
        self.output = self.output.reshape(self.batch_size, out_h, out_w, self.channels)

        return self.output

    def backward(self, gradient):
        gradient = gradient[:, :, cupy.newaxis, :, cupy.newaxis, :]
        gradient = gradient * self.mask
        gradient = gradient.reshape(self.batch_size, self.in_h, self.in_w, self.channels)
        return gradient
    

class AveragePool(Layer):

    def __init__(self, pool_height, pool_width):
        super(AveragePool, self).__init__()

        self.pool_h, self.pool_w = (pool_height, pool_width)
        self.batch_size, self.in_h, self.in_w, self.channels = 0,0,0,0
        self.mask = None

    def forward(self, input):

        self.input = input
        self.batch_size, self.in_h, self.in_w, self.channels = self.input.shape
        
        out_h = self.in_h // self.pool_h
        out_w = self.in_w // self.pool_w

        view = self.input[:, :out_h*self.pool_h, :out_w*self.pool_w, :]
        view = view.reshape(self.batch_size, out_h, self.pool_h, out_w, self.pool_w, self.channels)
        
        self.output = view.mean(axis=(2, 4), keepdims = True)

        self.mask   = cupy.ones(view.shape) / (self.pool_h * self.pool_w)
        self.output = self.output.reshape(self.batch_size, out_h, out_w, self.channels)

        return self.output

    def backward(self, gradient):
        gradient = gradient[:, :, cupy.newaxis, :, cupy.newaxis, :]
        gradient = gradient * self.mask
        gradient = gradient.reshape(self.batch_size, self.in_h, self.in_w, self.channels)
        return gradient


class Flatten(Layer):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        self.input = input
        self.output = self.input.reshape((self.input.shape[0], -1))
        return self.output

    def backward(self, gradient):
        return gradient.reshape(self.input.shape)
    
    
class Dense(Layer):

    def __init__(self, input_size, output_size):
        super(Dense, self).__init__()

        self.weights = init_random_tensor((input_size, output_size)) / input_size**0.5
        self.bias    = init_zeros_tensor(output_size)

        self.weight_grads   = init_zeros_tensor(self.weights.shape)
        self.weight_moments = init_zeros_tensor(self.weights.shape)
        self.weight_vars    = init_zeros_tensor(self.weights.shape)

        self.bias_grads   = init_zeros_tensor(self.bias.shape)
        self.bias_moments = init_zeros_tensor(self.bias.shape)
        self.bias_vars    = init_zeros_tensor(self.bias.shape)

        self.parameters = [self.weights,        self.bias]
        self.gradients  = [self.weight_grads,   self.bias_grads]
        self.moments    = [self.weight_moments, self.bias_moments]
        self.variances  = [self.weight_vars,    self.bias_vars]

    def forward(self, input):
        self.input = input
        self.output = input @ self.weights + self.bias
        return self.output

    def backward(self, gradient):
        self.weight_grads += (self.input.transpose() @ gradient) / gradient.shape[0]
        self.bias_grads += cupy.mean(gradient, (0))
        return gradient @ self.weights.transpose()


class Dropout(Layer):
    def __init__(self, dropout_rate):
        super(Dropout, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.dropout_rng = cupy.random.default_rng()
        self.droput_neurons = None
    
    def forward(self, input):
        self.input = input
        if self.dropout_rate > 0.0 and self.eval_mode is False:
            self.dropout_neurons = self.dropout_rng.random(self.input.shape, dtype = FLOAT_TYPE) >= self.dropout_rate
            self.dropout_neurons = self.dropout_neurons / FLOAT_TYPE(1 - self.dropout_rate)
            self.output = self.input * self.dropout_neurons
        else:
            self.output = self.input
        return self.output
        
    def backward(self, gradient):
        if self.dropout_rate > 0.0 and self.eval_mode is False:
            gradient *= self.dropout_neurons
        return gradient


class MultiHeadAttention(Layer):

    def __init__(self, embedding_dim, context_length, num_heads, decoder = False):
        super(MultiHeadAttention, self).__init__()

        self.embedding_dim  = embedding_dim
        self.context_length = context_length

        self.num_heads = num_heads
        self.heads_dim = embedding_dim // num_heads

        self.mask = cupy.tril(cupy.ones((context_length, context_length)))
        self.mask = cupy.where(self.mask == 0, -1e9, 0.0).astype(FLOAT_TYPE, copy = False)

        self.query = None
        self.key   = None
        self.value = None

        self.is_decoder = decoder

        self.qkv_weights = init_random_tensor((embedding_dim, 3*embedding_dim)) / embedding_dim**0.5
        self.out_weights = init_random_tensor((embedding_dim,   embedding_dim)) / embedding_dim**0.5

        self.qkv_weight_grads = init_zeros_tensor(self.qkv_weights.shape)
        self.qkv_weight_moments = init_zeros_tensor(self.qkv_weights.shape)
        self.qkv_weight_vars    = init_zeros_tensor(self.qkv_weights.shape)

        self.out_weight_grads = init_zeros_tensor(self.out_weights.shape)
        self.out_weight_moments = init_zeros_tensor(self.out_weights.shape)
        self.out_weight_vars    = init_zeros_tensor(self.out_weights.shape)

        self.parameters = [self.qkv_weights,        self.out_weights]
        self.gradients  = [self.qkv_weight_grads,   self.out_weight_grads]
        self.moments    = [self.qkv_weight_moments, self.out_weight_moments]
        self.variances  = [self.qkv_weight_vars,    self.out_weight_vars]


    def forward(self, input):

        self.input = input
        B, T, C    = self.input.shape

        qkv = self.input @ self.qkv_weights
        self.query, self.key_t, self.value = cupy.split(qkv, 3, axis = 2)

        self.query = self.query.reshape((B, T, self.num_heads, self.heads_dim)).transpose(0, 2, 1, 3)
        self.key_t = self.key_t.reshape((B, T, self.num_heads, self.heads_dim)).transpose(0, 2, 3, 1)
        self.value = self.value.reshape((B, T, self.num_heads, self.heads_dim)).transpose(0, 2, 1, 3)

        attends = (self.query @ self.key_t) / self.heads_dim**.5

        if self.is_decoder:
            attends += self.mask[:T,:T]

        normalization = cupy.max(attends, axis = -1, keepdims = True)
        exponent      = cupy.exp(attends - normalization)
        self.softmax  = exponent / cupy.sum(exponent, axis = -1, keepdims=True)

        self.heads_out = self.softmax @ self.value
        self.heads_out = self.heads_out.transpose(0, 2, 1, 3).reshape(B, T, C)

        self.output = self.heads_out @ self.out_weights

        return self.output

    def backward(self, gradient):

        B, T, C = gradient.shape

        self.out_weight_grads = cupy.tensordot(self.heads_out.transpose(2, 0, 1), gradient, 2) / B
        gradient = gradient @ self.out_weights.transpose()

        gradient = gradient.reshape((B, T, self.num_heads, self.heads_dim)).transpose(0, 2, 1, 3)

        value_grads = self.softmax.transpose(0, 1, 3, 2) @ gradient
        gradient = gradient @ self.value.transpose(0, 1, 3, 2)

        gradient = self.softmax * ( gradient - (gradient * self.softmax).sum(axis = -1, keepdims=True))
        gradient = gradient / self.heads_dim**.5

        key_t_grads = self.query.transpose(0, 1, 3, 2) @ gradient
        query_grads = gradient @ self.key_t.transpose(0, 1, 3, 2)

        query_grads = query_grads.transpose(0, 2, 1, 3).reshape(B, T, C)
        key_t_grads = key_t_grads.transpose(0, 3, 1, 2).reshape(B, T, C)
        value_grads = value_grads.transpose(0, 2, 1, 3).reshape(B, T, C)

        gradient = cupy.concatenate((query_grads, key_t_grads, value_grads), axis = 2)

        self.qkv_weight_grads += cupy.tensordot(self.input.transpose(2, 0, 1), gradient, 2) / B
        return gradient @ self.qkv_weights.transpose()


class ChannelFeedForward(Layer):

    def __init__(self, input_channels, output_channels, activation, dropout_rate = 0.0):
        super(ChannelFeedForward, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.activation:Layer = activation
        self.dropout = Dropout(dropout_rate)
        self.hidden_output = None

        self.weights1 = init_random_tensor((input_channels,  4*output_channels)) / input_channels**0.5
        self.bias1    = init_zeros_tensor(4*output_channels)

        self.weights2 = init_random_tensor((4*output_channels, output_channels)) / (4*output_channels)**0.5
        self.bias2    = init_zeros_tensor(output_channels)

        self.weight_grads1   = init_zeros_tensor(self.weights1.shape)
        self.weight_moments1 = init_zeros_tensor(self.weights1.shape)
        self.weight_vars1    = init_zeros_tensor(self.weights1.shape)

        self.bias_grads1   = init_zeros_tensor(self.bias1.shape)
        self.bias_moments1 = init_zeros_tensor(self.bias1.shape)
        self.bias_vars1    = init_zeros_tensor(self.bias1.shape)

        self.weight_grads2   = init_zeros_tensor(self.weights2.shape)
        self.weight_moments2 = init_zeros_tensor(self.weights2.shape)
        self.weight_vars2    = init_zeros_tensor(self.weights2.shape)

        self.bias_grads2   = init_zeros_tensor(self.bias2.shape)
        self.bias_moments2 = init_zeros_tensor(self.bias2.shape)
        self.bias_vars2    = init_zeros_tensor(self.bias2.shape)


        self.parameters = [self.weights1,        self.bias1,         self.weights2,        self.bias2]
        self.gradients  = [self.weight_grads1,   self.bias_grads1,   self.weight_grads2,   self.bias_grads2]
        self.moments    = [self.weight_moments1, self.bias_moments1, self.weight_moments2, self.bias_moments2]
        self.variances  = [self.weight_vars1,    self.bias_vars1,    self.weight_vars2,    self.bias_vars2]


    def forward(self, input):

        self.input = input

        self.hidden_output = self.activation.forward(self.input @ self.weights1 + self.bias1)
        self.hidden_output = self.dropout.forward(self.hidden_output)
        
        self.output = self.hidden_output @ self.weights2 + self.bias2

        return self.output

    def backward(self, gradient):

        self.weight_grads2 += cupy.tensordot(self.hidden_output.transpose(2, 0, 1), gradient, 2) / (gradient.shape[0])
        self.bias_grads2  += cupy.mean(gradient, (0, 1))
        
        gradient = self.dropout.backward(gradient @ self.weights2.transpose())
        gradient = self.activation.backward(gradient)

        self.weight_grads1 += cupy.tensordot(self.input.transpose(2, 0, 1), gradient, 2) / (gradient.shape[0])
        self.bias_grads1  += cupy.mean(gradient, (0, 1))

        return gradient @ self.weights1.transpose()

    def set_eval(self, eval_mode):
        self.dropout.set_eval(eval_mode)


class LayerNorm(Layer):

    def __init__(self, num_channels):
        super(LayerNorm, self).__init__()

        self.channels = num_channels
        self.eps = 1e-5

        self.gamma = cupy.ones(num_channels, dtype = FLOAT_TYPE)
        self.beta  = init_zeros_tensor(num_channels)

        self.gamma_grads   = init_zeros_tensor(num_channels)
        self.gamma_moments = init_zeros_tensor(num_channels)
        self.gamma_vars    = init_zeros_tensor(num_channels)

        self.beta_grads   = init_zeros_tensor(num_channels)
        self.beta_moments = init_zeros_tensor(num_channels)
        self.beta_vars    = init_zeros_tensor(num_channels)

        self.mean = None
        self.var = None
        self.std = None
        self.centered = None
        self.normed = None

        self.parameters = [self.gamma,         self.beta]
        self.gradients  = [self.gamma_grads,   self.beta_grads]
        self.moments    = [self.gamma_moments, self.beta_moments]
        self.variances  = [self.gamma_vars,    self.beta_vars]

    def forward(self, input):

        self.input = input

        self.mean = cupy.mean(input, axis=-1, keepdims=True)
        self.var  = cupy.var(input, axis=-1, keepdims=True)

        self.centered = input - self.mean

        self.std = (self.var + self.eps)**0.5
        self.normed = self.centered / self.std

        self.output = self.gamma * self.normed + self.beta
        return self.output

    def backward(self, gradient):

        self.gamma_grads += cupy.sum(gradient * self.normed, axis=(0, 1)) / gradient.shape[0]
        self.beta_grads  += cupy.sum(gradient, axis=(0, 1)) / gradient.shape[0]

        gradient = gradient * self.gamma
        
        gradient_normed = (gradient - cupy.mean(gradient, axis = -1, keepdims = True)) / self.std
        
        return gradient_normed - self.centered * (cupy.mean(gradient * self.centered, axis = -1, keepdims = True) / self.std**3)


class TransformerBlock(Layer):

    def __init__(self, embed_dim, context_length, num_heads, activation, decoder = False, dropout_rate = 0.0):
        super(TransformerBlock, self).__init__()

        self.pre_attn_norm = LayerNorm(embed_dim)
        self.attn_block = MultiHeadAttention(embed_dim, context_length, num_heads, decoder = decoder)

        self.pre_ffn_norm = LayerNorm(embed_dim)
        self.ffn = ChannelFeedForward(embed_dim, embed_dim, activation, dropout_rate = dropout_rate)

        self.parameters = [*self.pre_attn_norm.parameters, *self.attn_block.parameters, *self.pre_ffn_norm.parameters, *self.ffn.parameters]
        self.gradients  = [*self.pre_attn_norm.gradients,  *self.attn_block.gradients,  *self.pre_ffn_norm.gradients,  *self.ffn.gradients]
        self.moments    = [*self.pre_attn_norm.moments,    *self.attn_block.moments,    *self.pre_ffn_norm.moments,    *self.ffn.moments]
        self.variances  = [*self.pre_attn_norm.variances,  *self.attn_block.variances,  *self.pre_ffn_norm.variances,  *self.ffn.variances]

    def forward(self,input):

        self.input  = input
        self.output = self.input  + self.attn_block.forward(self.pre_attn_norm.forward(self.input))
        self.output = self.output + self.ffn.forward(self.pre_ffn_norm.forward(self.output))
        return self.output

    def backward(self, gradient):
        gradient += self.pre_ffn_norm.backward(self.ffn.backward(gradient))
        gradient += self.pre_attn_norm.backward(self.attn_block.backward(gradient))
        return gradient

    def set_eval(self, eval_mode):
        self.ffn.dropout.set_eval(eval_mode)