import cupy

global FLOAT_TYPE 
FLOAT_TYPE = cupy.float32 # (TF32 enabled)


def init_random_tensor(size):
    rng = cupy.random.default_rng()
    return rng.standard_normal(size, dtype = FLOAT_TYPE)

def init_zeros_tensor(size):
    return cupy.zeros(size, dtype = FLOAT_TYPE)

class Layer():

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