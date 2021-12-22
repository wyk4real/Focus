import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def back_projection_demo():
    sample = cv.imread(r'C:\Intro ML\openCV\image\8s.jpg')
    target = cv.imread(r'C:\Intro ML\openCV\image\8.jpg')
    sample_hsv = cv.cvtColor(sample,cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target,cv.COLOR_BGR2HSV)

    # cv.imshow('sample',sample)
    # cv.imshow('target',target)

    sample_hsv_Hist = cv.calcHist(sample_hsv,[0,1],None,[32,32],[0,180,0,256])
    cv.normalize(sample_hsv_Hist,sample_hsv_Hist,0,255,cv.NORM_MINMAX)
    dst = cv.calcBackProject([target_hsv],[0,1],sample_hsv_Hist,[0,180,0,256],1)
    cv.imshow('back_projection_demo',dst)


def hist2d_demo(image):
    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    hist = cv.calcHist(image,[0,1],None,[180,256],[0,180,0,256])
    plt.imshow(hist,interpolation='nearest')
    plt.title('2D Histogram')
    plt.show()




src = cv.imread(r'C:\Intro ML\openCV\image\4.jpg')
# cv.imshow('src',src)

# hist2d_demo(src)
back_projection_demo()

cv.waitKey(0)
cv.destroyAllWindows()
