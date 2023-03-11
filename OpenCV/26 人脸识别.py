import cv2 as cv
import numpy as np


def face_detect_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    face_detector = cv.CascadeClassifier('C:/Intro ML/openCV/data/haarcascades/haarcascade_frontalface_alt_tree.xml')
    faces = face_detector.detectMultiScale(gray, 1.01, 2)
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv.imshow('result', image)


src = cv.imread(r'C:\Intro ML\openCV\image\999.jpg')
src = cv.resize(src, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)  # 改变图片尺寸，原图像的0.4倍
cv.imshow('input image', src)

# 视频人脸检测
# capture = cv.VideoCapture(0)
# while(True):
#     ret,frame = capture.read()
#     frame = cv.flip(frame,1)
#     face_detect_demo(frame)
#     c = cv.waitKey(10)
#     if c == 27: #ESC键
#         break

face_detect_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
