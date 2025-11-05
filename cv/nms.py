# coding=utf-8
import numpy as np

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


def nms(dets, ths):
    # dets is numpy type
    scores = dets[:, 4]  # 取出预测分数
    # 此处的排序方法较为关键！！！
    order = scores.argsort()[::-1]  # ::-1表示逆序
    keep = []  # 保留的框对应的索引
    visited = [False] * len(order)  # 标记元素是否被访问过，默认为没有被访问过

    for i in range(len(order)):
        if visited[i]:
            pass
        else:
            index = order[i]  # 取出真正的目标索引
            keep.append(index)
            p1 = dets[index, :4]  # 取出对应的框
            j = i + 1  # 从下一个元素开始判断
            while j < len(order):
                if visited[j]:
                    pass
                else:
                    p2 = dets[order[j], :4]
                    iou_val = iou(p1, p2)
                    # 如果大于阈值说明为重叠的框，则将其标记为被访问过的状态
                    if iou_val >= ths:
                        visited[j] = True
                j += 1
    return keep


if __name__ == '__main__':
    # 简单 NMS 测试用例
    # 框格式: [x1, y1, x2, y2, score]
    dets = np.array([
        [0,   0,   100, 100, 0.9],  # A
        [10,  10,  110, 110, 0.8],  # B 与 A 有较大重叠，IoU≈0.68
        [200, 200, 300, 300, 0.7],  # C 远离 A/B
        [210, 210, 290, 290, 0.6],  # D 与 C 重叠，IoU≈0.64
    ], dtype=float)

    keep = nms(dets, ths=0.5)
    print('keep:', keep)
    assert keep == [0, 2], f"Expected [0, 2], got {keep}"
    print('NMS test passed.')
    