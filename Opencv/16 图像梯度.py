import cv2 as cv
import numpy as np
import sympy
from matplotlib import pyplot as plt


def sobel_demo(image): #索伯算子求一阶导数
    # grad_x = cv.Sobel(image,cv.CV_32F,1,0) #32位浮点数，在x方向上求导
    # grad_y = cv.Sobel(image,cv.CV_32F,0,1)

    grad_x = cv.Scharr(image,cv.CV_32F,1,0) #索伯算子的增强版，Scharr算子（边缘增强）
    grad_y = cv.Scharr(image,cv.CV_32F,0,1)

    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)

    gradx = cv.convertScaleAbs(grad_x) #求绝对值，然后转为8位,即0-255之间
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow('gradient_x',gradx)
    cv.imshow('gradient_y',grady)

    gradxy = cv.addWeighted(gradx,0.5,grady,0.5,0)
    cv.imshow('gradxy',gradxy)



def Laplace(image): #拉普拉斯算子
    dst = cv.Laplacian(image,cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow('Laplace',lpls)


def Laplace_self(image): #拉普拉斯算子自定义
    kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    dst = cv.filter2D(image,cv.CV_32F,kernel=kernel)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow('Laplace',lpls)


src = cv.imread(r'C:\Intro ML\openCV\image\5.jpg')
cv.imshow('src', src)

# sobel_demo(src)
# Laplace(src)
Laplace_self(src)

cv.waitKey(0)
cv.destroyAllWindows()
