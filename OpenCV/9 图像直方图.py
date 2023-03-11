import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def image_hist(image):
    color = ('b', 'g', 'r')
    for i, color in enumerate(color):
        hist = cv.calcHist(image, [i], None, [256], [0, 256])
        plt.plot(hist, color)
        plt.xlim([0, 256])  # 显示x轴的作图范围
    plt.show()


src = cv.imread(r'C:\Intro ML\openCV\image\11.jpg')
cv.imshow('src', src)

image_hist(src)

cv.waitKey(0)
cv.destroyAllWindows()
