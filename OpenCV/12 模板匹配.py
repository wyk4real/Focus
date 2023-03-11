import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def template():
    tpl = cv.imread(r'C:\Intro ML\openCV\image\2s.jpg')
    target = cv.imread(r'C:\Intro ML\openCV\image\2.jpg')
    cv.imshow('sample', tpl)
    cv.imshow('target', target)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]  # 三种方法，平方，相关性，相关性因子
    th, tw = tpl.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, tpl, md)  # 模板匹配函数，依次计算模板与待测图片的重叠区域的相似度，并将结果存入result当中
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)  # 返回result中最大值和最小值所对应的位置
        if md == cv.TM_CCORR_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = tl[0] + tw, tl[1] + th  # 计算红色小方块的宽和高
        cv.rectangle(target, tl, br, (0, 0, 255), 2)  # 在原图中显示小方格（红色小方格）
        cv.imshow('match' + str(md), target)
        # cv.imshow('match' + str(md), result) #输出计算的结果


template()

cv.waitKey(0)
cv.destroyAllWindows()
