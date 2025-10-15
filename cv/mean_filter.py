# coding=utf-8

import numpy as np

def mean_filter(img, ksize):
    padding = ksize // 2    # (F-k+2P)/S + 1 = F_
    h, w, _ = img.shape
    src = np.zeros((h + 2 * padding, w + 2 * padding))
    src[padding:h+padding, padding:w+padding] = img
    kernel = np.ones((ksize, ksize))
    dst = np.zeros(img.shape)

    for i in range(0, h):
        for j in range(0, w):
            dst[i, j] = np.sum(kernel * src[i:i+ksize][j:j+ksize]) // (ksize**2)
    return dst
