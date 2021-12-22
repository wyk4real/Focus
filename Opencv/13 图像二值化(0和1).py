import cv2 as cv
import numpy as np
import sympy
from matplotlib import pyplot as plt


def threshold_demo(image): #全局阈值
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU) #不给定阈值，给定方法(给不给阈值不起作用，如果有方法的前提下)
    # ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY) #给定阈值，不给定方法[与上面的方法二选一]
    # ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV) #取反
    # ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC) #截断，大于=127取127，小于不变
    # ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_TOZERO) #截断，大于=127不变，小于127取0
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)

    print('threshold_value:',ret)
    cv.imshow('binary',binary)


def local_threshold(image):  # 局部阈值
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # binary = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,25,10) #全部均值方法
    binary = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,25,10) #高斯均值方法,blocksize必须是奇数
    cv.imshow('binary',binary)


def custom_threshold(image):#自定义阈值即均值，计算均值
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h,w = gray.shape[:2]
    m = np.reshape(gray,[1,w*h]) #二维数组变为一位数组,1行 w+h列
    mean = m.sum() / (w*h)
    print('mean:',mean)
    ret,binary = cv.threshold(gray,mean,255,cv.THRESH_BINARY)
    cv.imshow('binary',binary)


src = cv.imread(r'C:\Intro ML\openCV\image\6.png')
cv.imshow('src',src)
 
# threshold_demo(src)
# local_threshold(src)
custom_threshold(src)

cv.waitKey(0)
cv.destroyAllWindows()
