# coding=utf-8
import numpy as np

def gray_image_max_value(img):
    # 统计每个灰度值出现的次数
    # 所有灰度值出现的次数总和等于图像中像素数的总数
    array = [0] * 256 

    h, w = img.shape[0], img.shape[1]
    for i in range(0, h):
        for j in range(0, w):
            array[img[i][j]] += 1
    
    max_value = -1
    min_value = -1
    for i in range(0, len(array)):
        if array[i] > 0:
            max_value = i
    
    for i in range(len(array)):
        if array[i] > 0:
            min_value = i
            break
    
    # 计算中值：需要考虑像素值的频次
    aa = []
    # 将统计的所有像素值按照其出现的次数添加到列表中
    for i in range(len(array)):
        aa.extend([i] * array[i])
    if len(aa) > 0:
        if len(aa) % 2 == 1:
            mid_value = aa[len(aa) // 2]
        else:
            mid_value = (aa[len(aa) // 2 - 1] + aa[len(aa) // 2]) / 2
    else:
        mid_value = -1
    
    return max_value, min_value, mid_value


if __name__ == '__main__':
    img = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    max_value, min_value, mid_value = gray_image_max_value(img)
    print(max_value, min_value, mid_value)
