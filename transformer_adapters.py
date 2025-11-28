import cupy

from utils import Layer, FLOAT_TYPE, init_random_tensor, init_zeros_tensor


class VitProjector(Layer):

    def __init__(self, input_size:tuple, patch_size:tuple, embedding_dim, num_latents = 0):
        super(VitProjector, self).__init__()

        self.batch_size = 0
        self.height, self.width, self.channels = input_size
        self.patch_size = patch_size

        self.patches_height = self.height // self.patch_size[0]
        self.patches_width  = self.width  // self.patch_size[1]

        self.sequence_length = self.patches_height * self.patches_width
        self.token_dim = self.patch_size[0] * self.patch_size[1] * self.channels
        self.embedding_dim = embedding_dim
        self.num_latents = num_latents

        self.tokens = None
        self.embeddings = None

        self.projection     = init_random_tensor((self.token_dim, self.embedding_dim)) / self.token_dim**0.5
        self.cls_lat_tokens = init_zeros_tensor((1 + self.num_latents, self.embedding_dim))

        pos = cupy.arange(self.sequence_length + 1 + self.num_latents, dtype = FLOAT_TYPE)[:, None]
        i   = cupy.arange(self.embedding_dim,                          dtype = FLOAT_TYPE)[None, :]

        self.positional_encoding = pos / 10000**(2 * (i // 2) / self.embedding_dim)
        self.positional_encoding[:, 0::2] = cupy.sin(self.positional_encoding[:, 0::2])
        self.positional_encoding[:, 1::2] = cupy.cos(self.positional_encoding[:, 1::2])

        self.projection_grads   = init_zeros_tensor(self.projection.shape)
        self.projection_moments = init_zeros_tensor(self.projection.shape)
        self.projection_vars    = init_zeros_tensor(self.projection.shape)
        
        self.cls_lat_token_grads   = init_zeros_tensor(self.cls_lat_tokens.shape)
        self.cls_lat_token_moments = init_zeros_tensor(self.cls_lat_tokens.shape)
        self.cls_lat_token_vars    = init_zeros_tensor(self.cls_lat_tokens.shape)

        self.parameters = [self.projection,         self.cls_lat_tokens]
        self.gradients  = [self.projection_grads,   self.cls_lat_token_grads]
        self.moments    = [self.projection_moments, self.cls_lat_token_moments]
        self.variances  = [self.projection_vars,    self.cls_lat_token_vars]

    def forward(self, input):

        self.input = input
        self.batch_size = self.input.shape[0]

        reshape = self.input.reshape((self.batch_size,
                                      self.patches_height, self.patch_size[0],
                                      self.patches_width, self.patch_size[1],
                                      self.channels))

        permute = reshape.transpose(0,1,3,2,4,5)
        self.tokens = permute.reshape(self.batch_size,
                                      self.sequence_length,
                                      self.token_dim)

        self.embeddings = self.tokens @ self.projection

        class_token_batch = cupy.tile(self.cls_lat_tokens, (self.batch_size, 1, 1))
        self.embeddings = cupy.concatenate([class_token_batch, self.embeddings], axis = 1)

        self.output = self.embeddings + self.positional_encoding
        return self.output

    def backward(self, gradient):

        self.cls_lat_token_grads += gradient[:,1 + self.num_latents,:].mean(axis = 0)
        gradient = gradient[:,1 + self.num_latents:,:]

        self.projection_grads += cupy.tensordot(self.tokens.transpose(2, 0, 1), gradient, 2) / gradient.shape[0]
        gradient = gradient @ self.projection.transpose()

        gradient = gradient.reshape((self.batch_size,
                                     self.patches_height,  self.patches_width,
                                     self.patch_size[0], self.patch_size[1], self.channels))

        gradient = gradient.transpose(0,1,3,2,4,5)
        gradient = gradient.reshape((self.batch_size, self.height, self.width, self.channels))
        return gradient


class VitMLPHead(Layer):

    def __init__(self, in_channels, out_channels):
        super(VitMLPHead, self).__init__()

        self.weights = init_random_tensor((in_channels, out_channels)) / in_channels**0.5
        self.bias    = init_zeros_tensor(out_channels)

        self.class_tokens = None

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
        self.class_tokens = input[:,0,:]
        self.output = self.class_tokens @ self.weights + self.bias

        return self.output

    def backward(self, gradient):

        self.weight_grads += (self.class_tokens.transpose() @ gradient) / gradient.shape[0]
        self.bias_grads += cupy.average(gradient, (0))

        gradient = gradient @ self.weights.transpose()
        gradient = gradient[:,cupy.newaxis,:]

        return cupy.concatenate((gradient, init_zeros_tensor(self.input.shape)[:,:-1,:]), axis = 1)


class GPTEmbedFront(Layer):

    def __init__(self, table, context_length):

        self.table         = table
        self.table_grads   = init_zeros_tensor(self.table.shape)
        self.table_moments = init_zeros_tensor(self.table.shape)
        self.table_vars    = init_zeros_tensor(self.table.shape)

        self.one_hot        = cupy.eye(table.shape[0], dtype=FLOAT_TYPE)
        self.one_hot_inputs = None

        pos = cupy.arange(context_length)[:, None]
        i   = cupy.arange(table.shape[1])[None, :]

        self.positional_encoding = pos / 10000**(2 * (i // 2) / table.shape[1])
        self.positional_encoding[:, 0::2] = cupy.sin(self.positional_encoding[:, 0::2])
        self.positional_encoding[:, 1::2] = cupy.cos(self.positional_encoding[:, 1::2])
        self.positional_encoding = self.positional_encoding.astype(FLOAT_TYPE, copy = False)

        self.parameters = [self.table]
        self.gradients  = [self.table_grads]
        self.moments    = [self.table_moments]
        self.variances  = [self.table_vars]

    def forward(self, input):
        self.input = input
        self.one_hot_inputs = self.one_hot[self.input]
        self.output = self.one_hot_inputs @ self.table + self.positional_encoding[:self.input.shape[1],:]
        return self.output

    def backward(self, gradient):
        self.table_grads += cupy.tensordot(self.one_hot_inputs.transpose(2, 0, 1), gradient, 2) / gradient.shape[0]
        return gradient @ self.table.transpose()


class GPTEmbedBack(Layer):

    def __init__(self, table):

        self.table         = table
        self.table_grads   = init_zeros_tensor(self.table.shape)
        self.table_moments = init_zeros_tensor(self.table.shape)
        self.table_vars    = init_zeros_tensor(self.table.shape)

        self.parameters = [self.table]
        self.gradients  = [self.table_grads]
        self.moments    = [self.table_moments]
        self.variances  = [self.table_vars]

    def forward(self, input):
        self.input = input
        self.output = self.input @ self.table.transpose()
        return self.output

    def backward(self, gradient):
        self.table_grads += cupy.tensordot(self.input.transpose(2, 0, 1), gradient, 2).transpose() / gradient.shape[0]
        return gradient @ self.table