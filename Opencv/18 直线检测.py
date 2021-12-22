import cv2 as cv
import numpy as np
import sympy
from matplotlib import pyplot as plt


def line_detection(image): #方法一
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150,apertureSize=3)
    lines = cv.HoughLines(edges,1,np.pi/180,200) #计算出每条线的rho和theta
    for line in lines:
        print(type(line))
        rho,theta = line[0]

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    cv.imshow('image_lines',image)



def line_detect_possible_demo(image): #方法二（常用方法）
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100,minLineLength=50,maxLineGap=10)
    for line in lines:
        print(type(line))
        x1,y1,x2,y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow('line_detect_possible_demo', image)



src = cv.imread(r'C:\Intro ML\openCV\image\7.jpg')
cv.imshow('src', src)

# line_detection(src)
line_detect_possible_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
