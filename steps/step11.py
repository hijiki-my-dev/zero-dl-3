from typing import Callable

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{data} is not supported")
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func: Callable):
        self.creator = func

    def backward(self):
        if self.grad is not None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, inputs: Variable):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)

        self.inputs: Variable = inputs
        self.outputs = outputs
        return outputs

    def forward(self, xs: list[np.ndarray]):
        raise NotImplementedError()

    def backward(self, gys: list[np.ndarray]):
        raise NotImplementedError


class Add(Function):
    def forward(self, xs: list[np.ndarray]):
        x0, x1 = xs
        y = x0 + x1
        return (y,)


class Square(Function):
    def forward(self, x: np.ndarray):
        return x**2
    
    def backward(self, gy: np.ndarray):
        x = self.input.data
        gx = 2*x*gy
        return gx


class Exp(Function):
    def forward(self, x: np.ndarray):
        y = np.exp(x)
        return y
    
    def backward(self, gy: np.ndarray):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)
