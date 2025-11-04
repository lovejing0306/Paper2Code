# coding=utf-8

def iou(p1, p2):  # 核心是重叠区域的坐标
    x11, y11, x12, y12 = p1
    x21, y21, x22, y22 = p2

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    # 计算重叠区域的坐标
    x1 = max(x11, x21)   # 从左上角的两个 x 值中取最大值
    y1 = max(y11, y21)   # 从左上角的两个 y 值中取最大值
    x2 = min(x12, x22)   # 从右下角的两个 x 值中取最小值
    y2 = min(y12, y22)   # 从右下角的两个 y 值中取最大值
    # 判断重叠区域是否存在
    if x1 < x2 and y1 < y2:
        ovr = (x2 - x1) * (y2 - y1)
        return ovr / (area1 + area2 - ovr)
    else:
        return 0.0


# 测试用例
if __name__ == "__main__":
    # 测试用例：两个有重叠的矩形框
    # 矩形1: (0, 0, 4, 4) - 面积为16
    # 矩形2: (2, 2, 6, 6) - 面积为16  
    # 重叠区域: (2, 2, 4, 4) - 面积为4
    # 预期IoU = 4 / (16 + 16 - 4) = 4/28 = 1/7 ≈ 0.1429
    
    box1 = (0, 0, 4, 4)
    box2 = (2, 2, 6, 6)
    
    result = iou(box1, box2)
    expected = 4.0 / 28.0  # 1/7
    
    print(f"矩形框1: {box1}")
    print(f"矩形框2: {box2}")
    print(f"计算得到的IoU: {result:.4f}")
    print(f"预期的IoU: {expected:.4f}")
    print(f"测试{'通过' if abs(result - expected) < 1e-6 else '失败'}")
    