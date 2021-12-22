import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract as tess


def recognize_text(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)

    kernal = cv.getStructuringElement(cv.MORPH_RECT,(1,2))
    bin1 = cv.morphologyEx(binary,cv.MORPH_OPEN,kernal)

    kernal = cv.getStructuringElement(cv.MORPH_RECT,(2,1))
    open_out = cv.morphologyEx(bin1,cv.MORPH_OPEN,kernal)
    cv.imshow('binary_image',open_out)


    #不能运行？？？？？？
    # cv.bitwise_not(open_out,open_out)
    # textImage = Image.fromarray(open_out)
    # text = tess.image_to_string(textImage)
    # print('识别结果：%s'%text)



src = cv.imread(r'C:\Intro ML\openCV\image\21.jpg')
cv.imshow('src', src)

recognize_text(src)

cv.waitKey(0)
cv.destroyAllWindows()
