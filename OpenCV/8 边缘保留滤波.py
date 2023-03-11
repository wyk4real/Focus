import cv2 as cv
import numpy as np


def bi_demo(image):  # 高斯双边滤波(边缘保留：像素差异比较大的地方)
    dst = cv.bilateralFilter(image, 0, 100, 15)
    cv.imshow('bi_demo', dst)


def shift_demo(image):  # 均值漂移滤波
    dst = cv.pyrMeanShiftFiltering(image, 10, 50)
    cv.imshow('shift_demo', dst)


src = cv.imread(r'C:\Intro ML\openCV\image\666.jpg')
cv.imshow('src', src)

# bi_demo(src)
shift_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
