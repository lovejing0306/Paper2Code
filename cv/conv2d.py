# coding=utf-8

def conv2d(image, kernel, padding=0, stride=1):
    """
    单通道图像的二维卷积操作
    
    参数:
        image: 输入图像 (H, W) - 二维列表或嵌套列表
        kernel: 卷积核 (K, K) - 二维列表
        padding: 填充大小，默认为0
        stride: 步长，默认为1
    
    返回:
        output: 卷积后的图像
    """
    # 获取图像和卷积核的尺寸
    image_h = len(image)
    image_w = len(image[0])
    kernel_h = len(kernel)
    kernel_w = len(kernel[0])
    
    # 对图像进行零填充
    if padding > 0:
        padded_image = []
        # 添加上方填充行
        for _ in range(padding):
            padded_image.append([0] * (image_w + 2 * padding))
        
        # 添加中间行（左右填充）
        for row in image:
            padded_row = [0] * padding + row + [0] * padding
            padded_image.append(padded_row)
        
        # 添加下方填充行
        for _ in range(padding):
            padded_image.append([0] * (image_w + 2 * padding))
        
        image = padded_image
        image_h = len(image)
        image_w = len(image[0])
    
    # 计算输出尺寸
    output_h = (image_h - kernel_h) // stride + 1  # 直接使用 padding 后的图像做计算输出特征图大小
    output_w = (image_w - kernel_w) // stride + 1
    
    # 初始化输出矩阵
    output = []
    for _ in range(output_h):
        output.append([0] * output_w)
    
    # 执行卷积操作
    for i in range(output_h):  # 以输出矩阵为参考
        for j in range(output_w):
            # 计算当前位置的卷积值
            conv_sum = 0
            for ki in range(kernel_h):
                for kj in range(kernel_w):
                    image_i = i * stride + ki
                    image_j = j * stride + kj
                    conv_sum += image[image_i][image_j] * kernel[ki][kj]
            
            output[i][j] = conv_sum
    
    return output


def conv2d_2(image, kernel, stride, padding):
    org_h = len(image)
    org_w = len(image[0])
    kernel_h = len(kernel)
    kernel_w = len(kernel[0])
    padding_h = org_h + 2*padding
    padding_w = org_w + 2*padding

    image_pad = []
    for _ in range(padding_h):
        image_pad.append([0] * padding_w)
    
    for r in range(org_h):
        for c in range(org_w):
            i = r + padding
            j = c + padding
            image_pad[i][j] = image[r][c]

    out_h = (org_h - kernel_h + 2*padding) // stride + 1
    out_w = (org_w - kernel_w + 2*padding) // stride + 1

    image_out = []
    for _ in range(out_h):
        image_out.append([0]*out_w)
    
    r_new = 0  # 标识新特征图在行方向的索引
    r_old = 0    # 标识原始特征图在行方向的起始索引
    while r_old <= padding_h - kernel_h:
        c_old = 0  # 标识原始特征图在列方向的起始索引
        c_new = 0   # 标识新特征图在列方向的起始索引
        while c_old <= padding_w - kernel_w:
            sum_ = 0
            for i in range(kernel_h):
                for j in range(kernel_w):
                    sum_ += image_pad[r_old + i][c_old + j] * kernel[i][j]
            image_out[r_new][c_new] = sum_
            c_old += stride  # 更新列索引
            c_new += 1   # 更新列所有呢
        r_old += stride  # 更新行索引
        r_new += 1     # 更新行索引
    return image_out
	

# 使用示例
if __name__ == "__main__":
    # 定义一个 5x5 的输入图像
    image = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ]
    
    # 定义一个 3x3 的卷积核（边缘检测算子）
    kernel = [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ]
    
    res2 = conv2d_2(image, kernel, padding=1, stride=1)
    print(res2)
    