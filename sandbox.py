# import torch
# from math import tanh

# from engine import Number
# from nn import Neuron, Linear, MLP
from test import Experiment


size = [4, 16, 128, 256, 256, 128, 4, 1]

exp = Experiment(
    size=size,
    bias=True,
    activation="tanh",
    copy2custom=1,
    copy2torch=0,
)

# mynet = exp.custom_MLP
# torchnet = exp.torch_MLP

# input = [0.34, 0.12, 1.53, 2.8]

exp_result = exp.run()
