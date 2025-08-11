import contextlib
from typing import Callable
import weakref

import numpy as np


class Config:
    enable_backprop = True


class Variable:
    def __init__(self, data: np.ndarray, name: str = None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{data} is not supported")
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " *9)
        return "variable(" + p + ")"

    def set_creator(self, func: Callable):
        self.creator = func
        self.generation = func.generation + 1

    def clear_grad(self):
        self.grad = None

    def backward(self, retain_grad: bool = False):
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
            gys = [output().grad for output in f.outputs] # 複数の出力がある場合、それをリストに格納
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

            # retain_gradがFalseの場合、途中の変数の微分をリセット（メモリ管理）
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None


class Function:
    def __call__(self, *inputs: list[Variable]):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)

            self.inputs: Variable = inputs
            self.outputs = [weakref.ref(output) for output in outputs] # 弱参照
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs: list[np.ndarray]):
        raise NotImplementedError()

    def backward(self, gys: list[np.ndarray]):
        raise NotImplementedError


# Numpyのように掛け算をできるようにする
class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray):
        y = x0 * x1
        return y

    def backward(self, gy: np.ndarray):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


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


@contextlib.contextmanager
def using_config(name: str, value):
    # with文の中でConfigクラスのname属性を書き換える
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        # with文を抜ける時に元に戻す
        setattr(Config, name, old_value)

def no_grad():
    return using_config("enable_backprop", False)

def mul(x0, x1):
    return Mul()(x0, x1)

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

# クラスの特殊メソッドに他の関数を指定

Variable.__mul__ = mul
Variable.__add__ = add

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

# y = add(mul(a, b), c)
y = a * b + c
y.backward()

print(y)
print(a.grad)
print(b.grad)
