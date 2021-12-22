import cv2 as cv
import numpy as np


def blur_demo(image): #均值滤波，（卷积）
    dst = cv.blur(image,(5,5))
    cv.imshow('blur_demo',dst)


def median_blur_demo(image): #中值滤波，（卷积）
    dst = cv.medianBlur(image,5)
    cv.imshow('median_blur_demo',dst)


def custon_blur_demo(image): #自定义滤波，（卷积）
    kernel = np.ones([5,5],np.float32)/25
    dst = cv.filter2D(image,-1,kernel)
    cv.imshow('custon_blur_demo',dst)

src = cv.imread(r'C:\Intro ML\openCV\image\11.jpg')
cv.imshow('src',src)

# blur_demo(src)
# median_blur_demo(src)
custon_blur_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()

