import cv2 as cv
import numpy as np


def measure_object(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 图像二值化
    print('threshold_value : %s' % ret)
    cv.imshow('binary image', binary)
    dst = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    contours, hireachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)  # 求的检测到的轮廓的面积
        x, y, w, h = cv.boundingRect(contour)  # 矩形边框，用一个最小的矩形，把找到的形状包起来。x，y是矩阵左上点的坐标，w，h是矩形的宽和高
        rate = min(w, h) / max(w, h)
        print('rectangle rate : %s' % rate)  # 宽高比
        mm = cv.moments(contour)  # 求原点矩
        print(type(mm))  # 字典类型
        cx = mm['m10'] / mm['m00']  # 得出质心坐标
        cy = mm['m01'] / mm['m00']  # 得出质心坐标

        cv.circle(image, (int(cx), int(cy)), 3, (0, 255, 255), -1)  # 画圆，thickness是正数，表示组成圆的线条的粗细程度.-1表示圆被填充
        # cv.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        print('contour area : %s' % area)  # 计算每个矩形的面积

        # approxCurve = cv.approxPolyDP(contour,4,True) #多边形逼近
        # print(approxCurve.shape)
        # if approxCurve.shape[0] > 0:
        #     cv.drawContours(dst,contours,i,(0,255,0),2)
        # if approxCurve.shape[0] > 4:
        #     cv.drawContours(dst, contours, i, (0, 255, 0), 2)
        # if approxCurve.shape[0] > 3:
        #     cv.drawContours(dst, contours, i, (0, 255, 0), 2)

    cv.imshow('measure_contours', image)


src1 = cv.imread(r'C:\Intro ML\openCV\image\19.jpg')
src = cv.resize(src1, (450, 500))  # 图片尺寸不能大于450*500
cv.imshow('src', src)

measure_object(src)

cv.waitKey(0)
cv.destroyAllWindows()
