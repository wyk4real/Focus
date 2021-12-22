import cv2 as cv
import numpy as np

def fill_color_demo(image): #彩色填充
    copyImg = image.copy()
    h,w = image.shape[:2]
    mask = np.zeros([h+2,w+2],dtype=np.uint8)
    cv.floodFill(copyImg,mask,(30,30),(0,255,255),(100,100,100),(50,50,50),flags=cv.FLOODFILL_FIXED_RANGE)
    cv.imshow('fill_color_demo',copyImg)


def fill_binary():
    image = np.zeros([400,400,3],dtype=np.uint8)
    image[100:300,100:300,:] = 255
    cv.imshow('fill_binary()',image)

    mask = np.ones([402,402,1],dtype=np.uint8)
    mask[101:301,101:301] = 0
    cv.floodFill(image,mask,(200,200),(0,0,255),cv.FLOODFILL_MASK_ONLY)
    cv.imshow('fill_binary',image)

fill_binary()


# src_1 = cv.imread(r'C:\Intro ML\openCV\image\11.jpg')
# src_2 = cv.imread(r'C:\Intro ML\openCV\image\22.jpg')
# src_3 = cv.imread(r'C:\Intro ML\openCV\image\4.jpg')
# cv.imshow('image',src_3)
#
# fill_color_demo(src_3)

# face = src_3[200:370,200:450] #ROI操作，取图片高和宽的具体位置
#
# gray = cv.cvtColor(face,cv.COLOR_BGR2GRAY) #把三通道rgb改为单通道
# print(gray.shape)
#
# black_face = cv.cvtColor(gray,cv.COLOR_GRAY2BGR) #把单通道改为三通道rgb
# print(black_face.shape)
# src_3[200:370,200:450] = black_face
#
# cv.imshow('NEW',src_3)


cv.waitKey(0)
cv.destroyAllWindows()