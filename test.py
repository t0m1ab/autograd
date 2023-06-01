from random import uniform
import torch
from time import time

from engine import Number
from nn import Linear, MLP


class TorchMLP(torch.nn.Module):
    
    def __init__(self, size: tuple, bias: bool, activation: str = "tanh"):
        super(TorchMLP, self).__init__()
        self.size = size
        self.num_layers = len(size) - 1
        self.bias = bias
        layers = []
        for k in range(len(size)-1):
            layers.append(torch.nn.Linear(size[k], size[k+1], bias=self.bias))
            if activation == "tanh":
                layers.append(torch.nn.Tanh())
        self.layers = torch.nn.ModuleList(layers)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class Experiment():
    """ Experiment object to test custom MLP and compare it to torch MLP implementations """

    def __init__(self, size: tuple, bias: bool, activation: float="tanh", copy2torch: bool=False, copy2custom: bool=False):
        self.size = size
        self.bias = bias
        self.torch_MLP = TorchMLP(size, bias, activation)
        self.custom_MLP = MLP(size, bias, activation)
        if copy2custom:
            self._copy2custom()
        elif copy2torch:
            self._copy2torch()
        else:
            print("WARNING: neural nets were initialized with different values")

    def _copy2torch(self):
        for idx, (custom_layer, torch_layer) in enumerate(zip(self.custom_MLP.layers, self.torch_MLP.layers)):

            if isinstance(custom_layer, Linear):
                my_weights = [[x.value for x in neuron.weights] for neuron in custom_layer.neurons]
                torch_layer.weight = torch.nn.Parameter(torch.tensor(my_weights))

                if self.bias:
                    my_biases = [neuron.bias.value for neuron in custom_layer.neurons]
                    torch_layer.bias = torch.nn.Parameter(torch.tensor(my_biases))
    
    def _copy2custom(self):
        for idx, (custom_layer, torch_layer) in enumerate(zip(self.custom_MLP.layers, self.torch_MLP.layers)):

            if isinstance(torch_layer, torch.nn.Linear):
                torch_weights = torch_layer.weight.tolist()
                for nidx, neuron in enumerate(custom_layer.neurons):
                    neuron.weights = [Number(x) for x in torch_weights[nidx]]

                if self.bias:
                    torch_bias = torch_layer.bias.tolist()
                    for nidx, neuron in enumerate(custom_layer.neurons):
                        neuron.bias.value = torch_bias[nidx]
 
    def check_num_parameters(self):

        # expected_count = 0
        # for dim1, dim2 in zip(self.size[:-1], self.size[1:]):
        #     expected_count += (dim1 + int(self.bias)) * dim2

        torch_count = self.torch_MLP.num_parameters()
        custom_count = self.custom_MLP.num_parameters()

        print("\n# NETWORKS")
        if torch_count != custom_count:
            raise ValueError(f"Number of parameters don't match: torch_count={torch_count} but custom_count={custom_count}")
        print(f"Number of parameters = {custom_count}")

    def forward_backward(self, input: list = None, verbose: bool = False):

        if self.size[-1] != 1:
            raise ValueError("Size[-1] != 1")

        if input is None:
            input = [uniform(-1,1) for _ in range(self.size[0])]
        elif not isinstance(input, list):
            raise ValueError(f"Input must be a list of float but received: {input}")

        if verbose:
            print("\nTORCH LAYERS:")
            for tlayer in self.torch_MLP.layers:
                print(tlayer.weight.tolist())

            print("\nCUSTOM LAYERS:")
            for clayer in self.custom_MLP.layers:
                print([[x.value for x in n.weights] for n in clayer.neurons])

        # forward pass torch
        torch_input = torch.tensor(list(input), requires_grad=True)
        torch_tic = time()
        torch_out = self.torch_MLP(torch_input)
        torch_tac = time()
        torch_out_value = torch_out.tolist()[0]
        # forward pass custom
        custom_input = [Number(x) for x in list(input)]
        custom_tic = time()
        custom_out = self.custom_MLP(custom_input)
        custom_tac = time()
        custom_out_value = custom_out.value

        print("\n# FORWARD")
        print(f"Custom output = {custom_out_value} | chrono = {custom_tac-custom_tic:.10f} seconds")
        print(f"Torch output = {torch_out_value} | chrono = {torch_tac-torch_tic:.10f} seconds")
        print(f"> Max absolute diff: {abs(torch_out_value - custom_out_value)}")

        # bacward pass torch
        torch_tic = time()
        torch_out.backward()
        torch_tac = time()
        # backward pass custom
        custom_tic = time()
        custom_out.backward()
        custom_tac = time()

        print("\n# BACKWARD")
        custom_input_grad = [x.grad for x in custom_input]
        torch_input_grad = torch_input.grad.tolist()
        print(f"Custom input gradients = {custom_input_grad[:2]} | chrono = {custom_tac-custom_tic:.10f} seconds")
        print(f"Torch input gradients = {torch_input_grad[:2]} | chrono = {torch_tac-torch_tic:.10f} seconds")
        max_diff = abs(torch_input_grad[0] - custom_input_grad[0])
        for x,y in zip(torch_input_grad, custom_input_grad):
            eps = abs(x-y)
            max_diff = eps if eps>max_diff else max_diff
        print(f"> Max absolute diff: {max_diff}")

        return max_diff < 5e-5
    
    def run(self):
        self.check_num_parameters()
        self.forward_backward()
        return True
    

if __name__ == "__main__":
    
    size = [4, 16, 128, 256, 256, 128, 4, 1]

    exp = Experiment(
        size=size,
        bias=True,
        activation="tanh",
        copy2custom=1,
        copy2torch=0,
    )

    exp_result = exp.run()