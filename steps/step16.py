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
        self.generation = 0

    def set_creator(self, func: Callable):
        self.creator = func
        self.generation = func.generation + 1

    def clear_grad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = []
        seen_set = set()
        def add_func(f: Function):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs] # 複数の出力がある場合、それをリストに格納
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)


class Function:
    def __call__(self, *inputs: list[Variable]):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])

        for output in outputs:
            output.set_creator(self)

        self.inputs: Variable = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs: list[np.ndarray]):
        raise NotImplementedError()

    def backward(self, gys: list[np.ndarray]):
        raise NotImplementedError


class Square(Function):
    def forward(self, x: np.ndarray):
        return x**2

    def backward(self, gy: np.ndarray):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


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

def add(x0, x1):
    return Add()(x0, x1)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()
print(y.data)
print(x.grad)
