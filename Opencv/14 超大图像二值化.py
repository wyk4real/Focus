import cv2 as cv
import numpy as np
import sympy
from matplotlib import pyplot as plt

def big_image_binary(image):
    print(image.shape)
    cw = ch = 128
    h,w = image.shape[:2]
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    for row in range(0,h,ch): #从0-h,间隔为ch
        for col in range(0,w,cw): #从0-w,间隔为cw
            roi = gray[row:row+ch,col:col+cw]
            dst = cv.adaptiveThreshold(roi,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,63,20)
            # ret,dst = cv.threshold(roi,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
            gray[row:row + ch, col:col + cw] = dst
            print(np.std(dst),np.mean(dst)) #计算每个小格子的方差和均值
    cv.imwrite(r'C:\Intro ML\openCV\image\result_binary.jpg',gray)





src = cv.imread(r'C:\Intro ML\openCV\image\Beispiel.jpg')
cv.imshow('src', src)

big_image_binary(src)

cv.waitKey(0)
cv.destroyAllWindows()
