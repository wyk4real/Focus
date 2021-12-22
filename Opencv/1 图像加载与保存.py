import cv2 as cv
import numpy as np

def get_image_info(image):
    print(type(image))
    print(image.shape)
    print(image.size)
    print(image.dtype)
    pixel_date = np.array(image)
    print(pixel_date)


def video_demo():
    capture = cv.VideoCapture(0)
    while(True):
        ret,frame = capture.read()
        frame=cv.flip(frame,1)
        cv.imshow('video',frame)
        c = cv.waitKey(50)
        if c == 27:
            break


src = cv.imread('C:\Intro ML\openCV\Beispiel.jpg')
cv.namedWindow('input image',cv.WINDOW_AUTOSIZE)
cv.imshow('input image',src)

get_image_info(src)

gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
cv.imwrite('C:\Intro ML\openCV\gray.png',gray)
cv.imshow('GRAY',gray)

video_demo()
cv.waitKey(0)
cv.destroyAllWindows()

























