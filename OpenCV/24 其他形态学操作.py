"""
顶帽：原图与开操作之间的差值图像
黑帽：闭操作与原图之间的差值图像

形态学梯度（基本梯度，内部梯度，外部梯度）
基本梯度：膨胀后的图像减去腐蚀后的图像得到的差值图像
内部梯度：原图减去腐蚀后的图像得到的差值图像
外部梯度：膨胀后的图像减去原图得到的差值图像
"""

import cv2 as cv
import numpy as np


def hat_gray_demo(image):  # 对灰度图像进行帽操作
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    dst = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)  # 顶帽
    # dst1 = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)#黑帽
    print(dst)

    # 增加图像亮度，可选
    cimage = np.array(gray.shape, np.uint8)
    cimage = 50
    print(cimage)
    dst = cv.add(dst, cimage)
    print(dst)

    cv.imshow('hat_gray_demo', dst)


def hat_binary_demo(image):  # 对二值图像进行帽操作
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow('binary', binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    dst = cv.morphologyEx(binary, cv.MORPH_TOPHAT, kernel)  # 顶帽
    # dst1 = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)#黑帽
    print(dst)

    # 增加图像亮度，可选
    cimage = np.array(gray.shape, np.uint8)
    cimage = 50
    print(cimage)
    dst = cv.add(dst, cimage)
    print(dst)

    cv.imshow('hat_binary_demo', dst)


def gradient_demo(image):  # 基本梯度
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow('binary', binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    dst = cv.morphologyEx(binary, cv.MORPH_GRADIENT, kernel)

    cv.imshow('gradient_demo', dst)


def gradient(image):  # 内外梯度

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dm = cv.dilate(image, kernel)
    em = cv.erode(image, kernel)
    dst1 = cv.subtract(image, em)  # 内部梯度
    dst2 = cv.subtract(dm, image)  # 外部梯度

    cv.imshow('gradient_innen', dst1)
    cv.imshow('gradient_außen', dst2)


src = cv.imread(r'C:\Intro ML\openCV\image\3.png')
cv.imshow('src', src)

# hat_gray_demo(src)
# hat_binary_demo(src)
# gradient_demo(src)
gradient(src)

cv.waitKey(0)
cv.destroyAllWindows()
