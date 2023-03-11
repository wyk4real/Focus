import cv2 as cv
import numpy as np


def watershed_demo(image):
    print(image.shape)

    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)  # 边缘保留滤波(必须噪声去除)

    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow('binary', binary)

    # 形态学操作
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)  # 连续俩次进行开操作
    sure_bg = cv.dilate(mb, kernel, iterations=3)
    cv.imshow('mor_opt', sure_bg)

    # 进行距离变换
    dist = cv.distanceTransform(mb, cv.DIST_L2, 3)
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    cv.imshow('distance_t', dist_output * 50)

    ret, surface = cv.threshold(dist, dist.max() * 0.6, 255, cv.THRESH_BINARY)
    cv.imshow('surface_bin', surface)

    surface_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg, surface_fg)
    ret, markers = cv.connectedComponents(surface_fg)
    print(ret)

    # 分水岭变换
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(src, markers=markers)
    src[markers == -1] = [0, 0, 255]
    cv.imshow('result', src)


src = cv.imread(r'C:\Intro ML\openCV\image\19.jpg')
cv.imshow('src', src)

watershed_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
