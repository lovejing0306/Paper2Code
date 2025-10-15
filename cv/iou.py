# coding=utf-8

def iou(p1, p2):  # 核心是重叠区域的坐标
    x11, y11, x12, y12 = p1
    x21, y21, x22, y22 = p2

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    # 计算重叠区域的坐标
    x1 = max(x11, x21)
    y1 = max(y11, y21)
    x2 = min(x12, x22)
    y2 = min(y12, y22)
    # 判断重叠区域是否存在
    if x1 < x2 and y1 < y2:
        ovr = (x2 - x1) * (y2 - y1)
        return ovr / (area1 + area2 - ovr)
    else:
        return 0.0
