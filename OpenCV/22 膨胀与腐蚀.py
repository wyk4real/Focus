import cv2 as cv
import numpy as np


def erode_demo(image):  # 图像腐蚀（减少）
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 转化为灰度图像
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 二值化图像
    cv.imshow('binary', binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (6, 6))  # 定义核函数，矩形3*3，数字越大，腐蚀越重
    dst = cv.erode(binary, kernel)  # 腐蚀，核函数3*3遍历图像，全为1取1否则取0（与）
    cv.imshow('erode_demo', dst)


def dilate_demo(image):  # 图像膨胀（增大）,特征更明显，与腐蚀相反
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 转化为灰度图像
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值化图像
    cv.imshow('binary', binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (6, 6))  # 定义核函数，矩形3*3，数字越大，腐蚀越重
    dst = cv.dilate(binary, kernel)  # 膨胀，核函数3*3遍历图像，全为0取0否则取1（或）
    cv.imshow('dilate_demo', dst)
    cv.waitKey()


def d_und_e(image):  # 图像膨胀（针对彩色图像）,最大值取代中心像素 / #图像腐蚀（针对彩色图像）,最小值取代中心像素

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (6, 6))  # 定义核函数，矩形3*3，数字越大，腐蚀越重
    # dst = cv.dilate(image,kernel) #膨胀，核函数3*3遍历图像，全为0取0否则取1（或）
    dst = cv.erode(image, kernel)  # 腐蚀，核函数3*3遍历图像，全为1取1否则取0（与）
    cv.imshow('dilate_demo', dst)


src = cv.imread('./1.jpg')
dilate_demo(src)
