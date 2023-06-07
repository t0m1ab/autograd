from math import tanh
from random import uniform


class Number():

    def __init__(self, value: str, children: tuple=()):
        self.value = value
        self.grad = 0
        self.children = children
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Number( {self.value} | grad = {self.grad} )"
    
    def __str__(self):
        return f"Number( {self.value} | grad = {self.grad} )"

    def __add__(self, other):
        other = other if isinstance(other, Number) else Number(other)
        out = Number(self.value + other.value, children=[self, other])

        def _backward():
            # a + b = c | dL/da = dL/dc * dc/da = dL/dc
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Number) else Number(other)
        out = Number(self.value * other.value, children=[self, other])

        def _backward():
            # a * b = c | dL/da = dL/dc * dc/da = dL/dc * b
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1 # well defined thanks to __mul__

    def __sub__(self, other):
        return self + (-other) # well defined thanks to __neg__

    def __rsub__(self, other):
        return other + (-self)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Number(self.value**other, children=[self])

        def _backward():
            self.grad += (other * self.value**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Number(0 if self.value < 0 else self.value, children=[self])

        def _backward():
            self.grad += (out.value > 0) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        out = Number(tanh(self.value), children=[self])

        def _backward():
            # tanh(x) = y | dL/dx = dL/dy * dy/dx = dL/dy * (1 - y**2)
            self.grad += out.grad * (1 - out.value**2)
        out._backward = _backward

        return out

    def topo_order(self):
        """ Order nodes of a DAG such that every node appears before all its children """
        order = []
        visited = set()
        def build_topo(node):  # invariant : a node is added iff all its children are already in order
            if node not in visited:
                visited.add(node)
                for v in node.children:
                    build_topo(v)
                order.append(node)
        build_topo(self)
        return reversed(order)

    def backward(self):
        self.grad = 1.0  # dL/dL = 1
        topo_order = self.topo_order()
        for node in topo_order:
            node._backward()