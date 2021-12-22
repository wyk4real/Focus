import cv2 as cv
import numpy as np
import sympy
from matplotlib import pyplot as plt


def pyramid_demo(image): #高斯金字塔
    level = 3
    temp = image.copy()
    pyramid_image = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_image.append(dst)
        cv.imshow('pyramid_dowm_'+str(i),dst)
        print(pyramid_image[i].shape)
        temp = dst.copy()
    return pyramid_image



def Laplace_demo(image): #拉普拉斯金字塔
    pyramid_image = pyramid_demo(image)
    level = len(pyramid_image)
    for i in range(level-1,-1,-1):
        if i == 0: #针对最后一个图像，必须找到原图才能计算
            expand = cv.pyrUp(pyramid_image[i],dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow('Laplace_demo' + str(i), lpls)

        else:
            expand = cv.pyrUp(pyramid_image[i],dstsize=pyramid_image[i-1].shape[:2]) #把小图变为和上一大图的尺寸
            lpls = cv.subtract(pyramid_image[i-1],expand) #用上一大图的尺寸减去它
            cv.imshow('Laplace_demo'+str(i),lpls)






src = cv.imread(r'C:\Intro ML\openCV\image\5.jpg')
cv.imshow('src', src)

# pyramid_demo(src) #处理图像金字塔的时候一定要注意，图像的尺寸必须是2的整数次方（尺寸调整），否则报错

Laplace_demo(src)


cv.waitKey(0)
cv.destroyAllWindows()
