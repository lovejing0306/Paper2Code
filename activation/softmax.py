# coding=utf-8
import numpy as np


def softmax(z):
    """Softmax 函数（用于输出层）"""
    max_value = np.max(z, axis=-1, keepdims=True)  # 求最大值
    exp_z = np.exp(z - max_value)  # 先对每个元素减去最大值，目的为了数值稳定性
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


if __name__ == '__main__':
    a = np.random.rand(5,4)
    res = softmax(a)
    print(res)