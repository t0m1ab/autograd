from random import uniform

from engine import Number


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        parameters = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                for neuron in layer.neurons:
                    parameters += neuron.weights
                    parameters.append(neuron.bias)
        return parameters

    def num_parameters(self):
        return len(self.parameters())


class Neuron(Module):

    def __init__(self, input_dim: int, bias: bool=True):
        super(Neuron, self).__init__()
        self.input_dim = input_dim
        self.is_bias = bias
        self.weights = [Number(uniform(-1,1)) for _ in range(input_dim)]
        if bias:
            self.bias = Number(uniform(-1,1))
        else:
            self.bias = Number(0)
    
    def __repr__(self):
        return f"Neuron( input_dim = {self.input_dim} | bias={self.is_bias})"

    def __call__(self, x):
        if len(x) != self.input_dim:
            raise ValueError(f"Neuron has input_dim={self.input_dim} but was called with data of size {len(x)}")

        out = sum([wi*xi for wi,xi in zip(self.weights,x)]) + self.bias
        
        return out


class Linear(Module):

    def __init__(self, input_dim, output_dim, bias: bool=True):
        super(Linear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.neurons = [Neuron(input_dim, bias=self.bias) for _ in range(output_dim)]

    def __repr__(self):
        return f"LinearLayer( input_dim = {self.input_dim} | output_dim={self.output_dim} | bias={self.bias})"

    def __call__(self, x):
        if len(x) != self.input_dim:
            raise ValueError(f"Layer has input_dim={self.input_dim} but was called with data of size {len(x)}")

        out = [n(x) for n in self.neurons]

        return out[0] if len(out) == 1 else out
    

class Tanh(Module):

    def __init__(self):
        super(Tanh, self).__init__()
        self.name = "tanh"

    def __repr__(self):
        return f"TanhActivation( ... )"

    def __call__(self, input):

        out = [x.tanh() for x in input] if isinstance(input, list) else input.tanh()

        return out


class MLP(Module):

    def __init__(self, size: tuple, bias: bool, activation: str = "tanh"):
        if len(size) == 0:
            raise ValueError("Cannot create empty MLP")
        super(MLP, self).__init__()
        self.size = size
        self.num_layers = len(size) - 1
        self.bias = bias
        self.layers = []
        for i in range(len(size)-1):
            self.layers.append(Linear(size[i], size[i+1], bias=self.bias))
            if activation == "tanh":
                self.layers.append(Tanh())
    
    def __repr__(self):
        return f"MultiLayerPerceptron( size = {self.size} | bias = {self.bias} | act_function = {self.act_function.name})"

    def __call__(self, x):
        if len(x) != self.size[0]:
            raise ValueError(f"MLP has input_dim={self.size[0]} but was called with data of size {len(x)}")

        for layer in self.layers:
            x = layer(x)
        
        return x