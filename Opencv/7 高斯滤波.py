import cv2 as cv
import numpy as np


def clamp(pv):
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv


def gauss_nosie(image):
    h,w,c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0,20,3) #在0-20之间生成3个随机数
            b = image[row,col,0]
            g = image[row,col,1]
            r = image[row,col,2]
            image[row,col,0] = clamp(b + s[0])
            image[row,col,1] = clamp(g + s[1])
            image[row,col,2] = clamp(r + s[2])
    cv.imshow('gauss_nosie',image)




src = cv.imread(r'C:\Intro ML\openCV\image\666.jpg')
cv.imshow('src',src)



# t1 = cv.getTickCount()
# gauss_nosie(src)
# t2 = cv.getTickCount()
# time = (t2-t1)/cv.getTickFrequency()
# print(time)

dst = cv.GaussianBlur(src,(5,5),15)  #高斯模糊API
cv.imshow('Gauss Blur',dst)

cv.waitKey(0)
cv.destroyAllWindows()

