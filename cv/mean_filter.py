# coding=utf-8

import numpy as np

def mean_filter(img, ksize):
    padding = ksize // 2    # (F-k+2P)/S + 1 = F_
    # h, w, _ = img.shape
    h, w = img.shape
    src = np.zeros((h + 2 * padding, w + 2 * padding))
    src[padding:h+padding, padding:w+padding] = img
    kernel = np.ones((ksize, ksize))
    dst = np.zeros(img.shape)

    for i in range(0, h):
        for j in range(0, w):
            dst[i, j] = np.sum(kernel * src[i:i+ksize, j:j+ksize]) // (ksize**2)
    return dst

if __name__ == '__main__':
    img = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=float,
    )

    # 期望输出：以 0 填充边界（padding=1），3x3 窗口的均值
    expected = np.array(
        [
            [1.33333333, 2.33333333, 1.77777778],
            [3.0,        5.0,        3.66666667],
            [2.66666667, 4.33333333, 3.11111111],
        ],
        dtype=float,
    )

    res = mean_filter(img, ksize=3)
    print(res)