import cupy
import numpy as np
import cv2

global FLOAT_TYPE 
FLOAT_TYPE = cupy.float32 # (TF32 enabled)


# inference mode: build a model with no training state at all

INFERENCE_MODE = False


class inference_mode:

    """Build and run a model without any of the state that only backward needs.

    Layers constructed inside this context allocate no gradient, moment or variance
    buffers - three extra full size copies of every parameter - and Network._forward
    drops each layer's cached activations as soon as the next layer has consumed them.
    For a Gemma 4 31B configuration that is the difference between ~474 GiB and
    ~115 GiB.

    Construction has to happen inside the context, because the buffers are allocated
    in each layer's __init__:

        with inference_mode():
            model = gemma_gpt(**config)

        model.predict(tokens)          # fine, inside or outside the context

    Layers remember how they were built, so prediction behaves correctly either way.
    Calling backward() on such a model raises, which is the intent - there is nowhere
    for a gradient to accumulate.
    """

    def __init__(self, enabled = True):
        self.enabled = enabled

    def __enter__(self):
        global INFERENCE_MODE
        self.previous  = INFERENCE_MODE
        INFERENCE_MODE = self.enabled
        return self

    def __exit__(self, *exception):
        global INFERENCE_MODE
        INFERENCE_MODE = self.previous
        return False


# scattered accumulation, used for embedding table gradients

try:
    from cupyx import scatter_add as _scatter_add
except ImportError: # running on CPU with numpy substituted for cupy
    def _scatter_add(target, indices, values):
        cupy.add.at(target, indices, values)

def scatter_add(target, indices, values):
    """In place target[indices] += values, summing contributions of repeated indices.

    Used instead of a one hot matmul for embedding lookups: the one hot route costs
    a (batch, sequence, vocab) intermediate, which is fine at char level vocabs and
    fatal at the 262144 token vocab a Gemma style model uses."""
    _scatter_add(target, indices, values)


# tensor initialization with float type

def init_random_tensor(size):
    rng = cupy.random.default_rng()
    return rng.standard_normal(size, dtype = FLOAT_TYPE)

def init_zeros_tensor(size):
    return cupy.zeros(size, dtype = FLOAT_TYPE)


# layer interface and residual layer wrapper

class Layer:

    # activations stashed by forward for backward to consume. Listed per layer so
    # clear_cache knows what is safe to drop: it must never touch anything the next
    # forward still needs, such as BatchNorm's running statistics or a cached mask.
    CACHED = ("input", "output")

    def __init__(self):

        self.input = None
        self.output = None

        self.parameters = []
        self.gradients = []
        self.moments = []
        self.variances = []

        self.eval_mode = False
        self.inference_only = INFERENCE_MODE

    def register(self, parameter):

        """Register a parameter and allocate its optimizer buffers.

        Returns the gradient buffer, so layers can keep a named handle on it. Under
        inference mode nothing is allocated and None is returned: the four lists stay
        parallel because they all stay empty, which quietly makes zero_grad and the
        optimizer step no-ops, and makes backward fail loudly on the None.
        """

        self.parameters.append(parameter)

        if self.inference_only:
            return None

        gradient = init_zeros_tensor(parameter.shape)

        self.gradients.append(gradient)
        self.moments.append(init_zeros_tensor(parameter.shape))
        self.variances.append(init_zeros_tensor(parameter.shape))

        return gradient

    def clear_cache(self):
        for name in self.CACHED:
            setattr(self, name, None)

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *input):
        raise NotImplementedError

    def set_eval(self, eval_mode):
        self.eval_mode = eval_mode

    def zero_grad(self):
        for grad in self.gradients:
            grad.fill(0)
    
    def zero_adam(self):
        for moment, variance in zip(self.moments, self.variances):
            moment.fill(0)
            variance.fill(0)
            

class Residual(Layer):
    
    def __init__(self, layers:list[Layer], mode = "add", concat_axis = 1):
        super(Residual, self).__init__()
        
        self.layers = layers
        self.mode = mode
        assert self.mode in {"add", "concat"}
        self.concat_axis = concat_axis
        
        self.parameters = []
        self.gradients  = []
        self.moments    = []
        self.variances  = []
        
        for layer in layers:
            self.parameters = self.parameters + layer.parameters
            self.gradients  = self.gradients  + layer.gradients
            self.moments    = self.moments    + layer.moments
            self.variances  = self.variances  + layer.variances
        
    def forward(self, input):
        
        self.input = input
        x = input
        
        for layer in self.layers:
            x = layer.forward(x)
        
        if self.mode == "add":
            self.output = self.input + x
        elif self.mode == "concat":
            self.output = cupy.concatenate((self.input, x), axis = self.concat_axis)
        return self.output
    
    def backward(self, gradient):
        
        gradient, nabla = (gradient, gradient) if self.mode == "add" else cupy.array_split(gradient, 
                                                                                           (self.input.shape[self.concat_axis], ),
                                                                                           axis = self.concat_axis)
        
        for layer in reversed(self.layers):
            nabla = layer.backward(nabla)
            
        gradient += nabla
        return gradient

    def set_eval(self, eval_mode):
        for layer in self.layers:
            layer.set_eval(eval_mode)

    def clear_cache(self):
        super(Residual, self).clear_cache()
        for layer in self.layers:
            layer.clear_cache()
            

# crude augments from scratch

def random_flip(data):
    indices = np.random.choice(np.arange(0, len(data)), size = len(data) // 2)
    data[indices] = np.flip(data[indices], axis = 3)
    return data
    
def random_shift(data):
    
    x_shifts = np.random.choice(np.arange(0,9),  size = len(data))
    y_shifts = np.random.choice(np.arange(0,9),  size = len(data))
    
    x_res = data.shape[2]
    y_res = data.shape[3]

    for i, (img, dx, dy) in enumerate(zip(data, x_shifts, y_shifts)):
        cropped = np.pad(img, ((0,0), (4,4), (4,4)))
        cropped = cropped[:, dx:x_res+dx, dy:y_res+dy]
        data[i] = cropped
    
    return data


def random_rotate(data):
    
    x_res = data.shape[2]
    y_res = data.shape[3]
    
    mid = (x_res // 2, y_res // 2)

    angles = np.random.randint(-15, 16, len(data))
    
    for i, (img, angle) in enumerate(zip(data, angles)):
        img = img.transpose(1, 2, 0)
        matrix  = cv2.getRotationMatrix2D(mid, angle, 1.0)
        data[i] = cv2.warpAffine(img, matrix, (x_res, y_res)).transpose(2, 0, 1)
    
    return data


def augment_images(data):
    x = data.copy()
    return random_shift(random_rotate(random_flip(x)))