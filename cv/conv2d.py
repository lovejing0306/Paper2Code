# coding=utf-8

def convolve2d(image, kernel, padding=0, stride=1):
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
    output_h = (image_h - kernel_h) // stride + 1
    output_w = (image_w - kernel_w) // stride + 1
    
    # 初始化输出矩阵
    output = []
    for _ in range(output_h):
        output.append([0] * output_w)
    
    # 执行卷积操作
    for i in range(output_h):
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
    
    # 执行卷积（无填充，步长为1）
    result = convolve2d(image, kernel, padding=0, stride=1)
    
    print("输入图像:")
    for row in image:
        print(row)
    
    print("\n卷积核:")
    for row in kernel:
        print(row)
    
    print("\n卷积结果:")
    for row in result:
        print(row)
    
    # 使用填充的例子
    print("\n\n使用 padding=1 的卷积结果:")
    result_padded = convolve2d(image, kernel, padding=1, stride=1)
    for row in result_padded:
        print(row)