from typing import Callable

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func: Callable):
        self.creator = func

    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input # 計算グラフの、1つ前の入力を取得
            x.grad = f.backward(self.grad)
            x.backward()


class Function:
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input: Variable = input
        self.output = output
        return output

    def forward(self, x: np.ndarray):
        raise NotImplementedError()

    def backward(self, gy: np.ndarray):
        raise NotImplementedError


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

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a


# y.grad = np.array(1.0)
# C = y.creator
# b = C.input
# b.grad = C.backward(y.grad)

# B = b.creator
# a = B.input
# a.grad = B.backward(b.grad)

# A = a.creator
# x = A.input
# x.grad = A.backward(a.grad)
# print(x.grad)

y.grad = np.array(1.0)
y.backward()
print(x.grad)
