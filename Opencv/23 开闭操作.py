import cv2 as cv
import numpy as np


def open_demo(image):#开操作，先腐蚀再膨胀(去除某些小的形状)
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow('binary',binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))#设置矩形的大小可以提取水平（15，1）和竖直的线条（1，15）
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))#设置不同的图形形状（椭圆）去遍历整个图像
    binary = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel) #单独的API
    cv.imshow('open_result',binary)


def close_demo(image):#闭操作，先膨胀再腐蚀(填充某些小的空缺)
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow('binary',binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    binary = cv.morphologyEx(binary,cv.MORPH_CLOSE,kernel) #单独的API
    cv.imshow('close_result',binary)



src = cv.imread(r'C:\Intro ML\openCV\image\23.png')
cv.imshow('src', src)

open_demo(src)
# close_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()