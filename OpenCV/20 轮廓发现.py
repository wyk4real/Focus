import cv2 as cv
import numpy as np
import sympy
from matplotlib import pyplot as plt


def contours_demo(image):  # 方法一：灰度，二值化，轮廓寻找，轮廓绘制
    dst = cv.GaussianBlur(image, (3, 3), 0)  # 高斯滤波，除去噪声
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)  # 转化为灰度图像
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值化图像
    cv.imshow('binary image', binary)
    contours, heriachy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
    for i, contour in enumerate(contours):  # i=0,1,2,3...
        cv.drawContours(image, contours, i, (0, 0, 255), 2)  # 画出轮廓，如果要填充轮廓，让2 = -1
        print(i)
    cv.imshow('detect_contours', image)


def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)  # 高斯模糊，去噪声

    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)  # 转化为灰度图像，非二值
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)

    edge_output = cv.Canny(gray, 50, 150)  # 直接对二值图像边缘提取，高低阈值3:1 or 2:1
    cv.imshow('Canny Edge', edge_output)
    return edge_output  # 为了后期调用，一定要返回值


def contours_demo_1(image):  # 方法二：灰度，边缘检测，轮廓寻找，轮廓绘制

    binary = edge_demo(image)  # 调用上面的边缘检测函数

    contours, heriachy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
    for i, contour in enumerate(contours):  # i=0,1,2,3...
        cv.drawContours(image, contours, i, (0, 0, 255), 2)  # 画出轮廓，如果要填充轮廓，让2 = -1
        print(i)
    cv.imshow('detect_contours', image)


src = cv.imread(r'C:\Intro ML\openCV\image\19.jpg')
cv.imshow('src', src)

contours_demo(src)
# contours_demo_1(src)

cv.waitKey(0)
cv.destroyAllWindows()
