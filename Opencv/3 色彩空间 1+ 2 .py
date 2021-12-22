import cv2 as cv
import numpy as np

# H(0-180) S(0-255) V(0-255)
def extrace_objekt_demo():
    capture = cv.VideoCapture('C:\Intro ML\openCV\DEMO.mp4') #视频读取
    while(True):
        ret,frame = capture.read()
        if ret == False:
            break

        hsv = cv.cvtColor(frame,cv.COLOR_RGB2HSV) #rgb色彩转化为hsv
        low_hsv = np.array([37,43,46])
        high_hsv = np.array([77,255,255])
        mask = cv.inRange(hsv,low_hsv,high_hsv)  #inRange过滤出绿色，查hsv表找到绿色值
        cv.imshow('mask',mask)

        cv.imshow('DEMO',frame)
        c = cv.waitKey(40)
        if c == 27:
            break



def color_space_demo(image):   #色彩转化 RGB 到 HSV YUV HIS YCrCb
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    cv.imshow('GRAY',gray)
    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    cv.imshow('HSV',hsv)
    # yuv = cv.cvtColor(image,cv.COLOR_BGR2YUV)
    # cv.imshow('YUV',yuv)



src = cv.imread('C:\Intro ML\openCV\Beispiel.jpg')
cv.namedWindow('input image',cv.WINDOW_AUTOSIZE)

# cv.imshow('input image',src)
#
color_space_demo(src)
b,g,r = cv.split(src) #通道分离
# cv.imshow('blue',b)
# cv.imshow('green',g)
# cv.imshow('red',r)

src[:,:,2] = 0 #rgb三通道，使得第三个通道的值为0
# cv.imshow('11',src)

src = cv.merge([r,g,b]) #通道合并
cv.imshow('22',src)

# extrace_objekt_demo()

cv.waitKey(0)
cv.destroyAllWindows()

























