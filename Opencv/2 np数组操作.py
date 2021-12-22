import cv2 as cv
import numpy
import numpy as np

def access_pixels(image):


    print(image.shape)     #输出图像的尺寸以及色彩的通道
    h = image.shape[0]
    w = image.shape[1]
    channels = image.shape[2]
    print('宽:%s,高:%s，通道:%s'%(h,w,channels))

    for row in range(h):
        for col in range(w):
            for c in range(channels):
                pv = image[row,col,c]    #计算每个小方格的色彩值
                image[row,col,c] = 255 - pv  #改变每个小方格的色彩值
    cv.imshow('new',image)


def inverse(image):
    src = cv.bitwise_not(image)
    cv.imshow('取反新方法',src)


def create_image():  #创建新图像-3通道
    img = np.zeros([400,400,3],np.uint8)
    img[:,:,0] = np.ones([400,400])*255
    cv.imshow('NEW',img)

# def create_image():  #创建新图像-单通道
#     img = np.zeros([400,400],np.uint8)
#     img[:,:] = np.ones([400,400])*255
#     cv.imshow('NEW', img)
# m=np.ones([3,3],np.uint8) #生成一个3*3的矩阵，数据类型是8位。
# m.fill(12222)
# print(m)
# mm=m.reshape([1,9])
# print(mm)


src = cv.imread('C:\Intro ML\openCV\Beispiel.jpg')     #三原色 rgb
src = cv.resize(src, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)  # 改变图片尺寸，原图像的0.4倍

cv.namedWindow('input image',cv.WINDOW_AUTOSIZE)
cv.imshow('input image',src)

t1 = cv.getTickCount()
# access_pixels(src)
# create_image()
inverse(src)

t2 = cv.getTickCount()
print('Time:%s ms'%((t2-t1)/cv.getTickFrequency()))
cv.waitKey(0)
cv.destroyAllWindows()
































