import cupy

global FLOAT_TYPE 
FLOAT_TYPE = cupy.float32 # (TF32 enabled)


def init_random_tensor(size):
    rng = cupy.random.default_rng()
    return rng.standard_normal(size, dtype = FLOAT_TYPE)

def init_zeros_tensor(size):
    return cupy.zeros(size, dtype = FLOAT_TYPE)

class Layer:

    def __init__(self):

        self.input = None
        self.output = None

        self.parameters = []
        self.gradients = []
        self.moments = []
        self.variances = []

        self.eval_mode = False

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *input):
        raise NotImplementedError

    def set_eval(self, eval_mode):
        self.eval_mode = eval_mode

    def zero_grad(self):
        for grad, moment, variance in zip(self.gradients, self.moments, self.variances):
            grad *= 0
            moment *= 0
            variance *= 0
            

class Residual(Layer):
    
    def __init__(self, layers:list[Layer], mode = "add"):
        super(Residual, self).__init__()
        
        self.layers = layers
        self.mode = mode
        assert self.mode in {"add", "concat"}
        
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
            self.output = cupy.concatenate(self.input, x, axis = -1)
        return self.output
    
    def backward(self, gradient):
        
        nabla    = gradient if self.mode == "add" else gradient[:,:,:,self.input.shape[-1]:]
        gradient = gradient if self.mode == "add" else gradient[:,:,:,:self.input.shape[-1]]
        
        for layer in reversed(self.layers):
            nabla = layer.backward(nabla)
            
        gradient += nabla
        return gradient

    def set_eval(self, eval_mode):
        for layer in self.layers:
            layer.set_eval(eval_mode)