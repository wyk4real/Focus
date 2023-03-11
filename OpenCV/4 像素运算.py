import cv2 as cv
import numpy as np


def add(m1, m2):  # 加法
    dest = cv.add(m1, m2)
    cv.imshow('add', dest)


def sub(m1, m2):  # 减法
    dest = cv.subtract(m1, m2)
    cv.imshow('sub', dest)


def mul(m1, m2):  # 乘法
    dest = cv.multiply(m1, m2)
    cv.imshow('mul', dest)


def div(m1, m2):  # 除法
    dest = cv.divide(m1, m2)
    cv.imshow('div', dest)


def others(m):  # 求均值
    M = cv.mean(m)  # 求均值，显示3个通道分别的均值
    MM = cv.meanStdDev(m)  # 求均值方差，显示3个通道分别的均值和方差，方差越大，说明颜色的差异度越大
    print(M)
    print(MM)


def logic_and(m1, m2):  # 逻辑与
    dest = cv.bitwise_and(m1, m2)
    cv.imshow('loic_add', dest)


def logic_or(m):  # 逻辑或
    dest = cv.bitwise_not(m)
    cv.imshow('logic_or', dest)


def logic_not(m1, m2):  # 逻辑非,一个变量
    dest = cv.bitwise_not(m1, m2)
    cv.imshow('logic_not', dest)
    print(dest.shape)


def contrast_brightness_demo(image, c, b):  # 改变图片的对比度和亮度 c为对比度，b为亮度
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1 - c, b)
    cv.imshow('contrast_brightness_demo', dst)


src_1 = cv.imread(r'C:\Intro ML\openCV\image\11.jpg')
src_2 = cv.imread(r'C:\Intro ML\openCV\image\22.jpg')

# print(src_1.shape)
# print(src_2.shape)

# cv.imshow('image1',src_1)
# cv.imshow('image2',src_2)
# add(src_1,src_2)
# sub(src_1,src_2)
# mul(src_1,src_2)
# div(src_1,src_2)
# others(src_1)
# logic_and(src_1,src_2)
# logic_or(src_1,src_2)
# logic_not(src_1)
contrast_brightness_demo(src_1, 1, 0)  # 对比度和亮度不变

cv.waitKey(0)
cv.destroyAllWindows()
