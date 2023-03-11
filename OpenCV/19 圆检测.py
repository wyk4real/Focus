import cv2 as cv
import numpy as np
import sympy
from matplotlib import pyplot as plt


def detect_circles_demo(image):
    dst = cv.pyrMeanShiftFiltering(image, 10, 100)  # 边缘保留滤波EPF,去除噪声
    B_image = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)  # 转为二值图像
    circles = cv.HoughCircles(B_image, cv.HOUGH_GRADIENT, 1, 50, param1=50, param2=20, minRadius=0, maxRadius=0)  # 圆检测
    circles = np.uint16(np.around(circles))  # 转化为整数,每一行为三个值，分别为圆心和半径
    print(circles)
    for i in circles[0, :]:
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 画圆
        cv.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)  # 画圆心，图中中心的蓝色小点，可画可不画
    cv.imshow('detect_circles_demo', image)


src = cv.imread(r'C:\Intro ML\openCV\image\19.jpg')
cv.imshow('src', src)

detect_circles_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
