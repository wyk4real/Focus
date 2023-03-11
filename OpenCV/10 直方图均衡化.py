import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def equalHist_demo(image):#全局直方图均衡化
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY) #直方图均衡化针对2值图像,对比度增强
    dst = cv.equalizeHist(gray)
    cv.imshow('equalHist_demo',dst)


def clahe_demo(image): #自适应直方图均衡化
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY) #直方图均衡化针对2值图像,对比度增强
    clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    dst = clahe.apply(gray)
    cv.imshow('clahe_demo',dst)
print('---------------------------------------------------------------------------------------------------------------')


def creat_rgb_hist(image):  # 自定义直方图
    h,w,c = image.shape
    rgbHist = np.zeros([16*16*16,1],np.float32)
    bsize = 256/16 #单个通道256个图像值分为16段，bsize就是每段的图象值范围
    for row in range(h):
        for col in range(w):
            b = image[row,col,0]
            g = image[row,col,1]
            r = image[row,col,2]
            index = int(b/bsize)*16*16 + int(g/bsize)*16 + int(r/bsize)
            rgbHist[int(index),0] = rgbHist[int(index),0] + 1
    return rgbHist



def hist_compare(image1,image2): #直方图比较
    hist1 = creat_rgb_hist(image1)
    hist2 = creat_rgb_hist(image2)
    match1 = cv.compareHist(hist1,hist2,cv.HISTCMP_BHATTACHARYYA) #巴式比较
    match2 = cv.compareHist(hist1,hist2,cv.HISTCMP_CORREL) #相关性比较
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR) #卡方比较
    print('巴氏:%s，相关性:%s，卡方:%s'%(match1,match2,match3))



src = cv.imread(r'C:\Intro ML\openCV\image\7.jpg')
# cv.imshow('src',src)

# equalHist_demo(src)
# clahe_demo(src)

src1 = cv.imread(r'C:\Intro ML\openCV\image\11.jpg')
src2 = cv.imread(r'C:\Intro ML\openCV\image\22.jpg')
hist_compare(src1,src2)


cv.waitKey(0)
cv.destroyAllWindows()
