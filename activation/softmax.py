# coding=utf-8
import numpy as np


def softmax(z):
    """Softmax 函数（用于输出层）"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 数值稳定性
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

