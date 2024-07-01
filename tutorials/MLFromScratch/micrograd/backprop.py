from typing import Tuple, Union
import numpy as np


class Value:

    def __init__(self, data: Union[int, float], children: Tuple = (), _op: str ='') -> None:
        self.data = data
        self.grad = 0.0

        self._prev = set(children)
        self._backward = lambda: None
        self._op = _op


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out


    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out


    def __pow__(self, other):
        isinstance(other, (int, float)), "only supporting int/float powers atm"
        out = Value(self.data ** other, (self, other), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out


    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return other * self

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def relu(self):
        #  val = self.data if self.data > 0 else 0
        val = max(0, self.data)
        out = Value(val, (self,), 'ReLu')

        def _backward():
            self.grad +=  (out.data > 0) * out.grad
        out._backward = _backward

        return out


    def tanh(self):
        num = np.exp(2 * self.data) - 1
        denom = np.exp(2 * self.data ) + 1
        val = num / denom
        out = Value(val, (self,), 'tanh')

        def _backward():
            self.grad += (1 - val**2) * out.grad
        out._backward = _backward

        return out


    def backward(self):
        """
        applying backprop to each weight in reversed topological order
        """
        # topological order
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # execute backprop
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()




