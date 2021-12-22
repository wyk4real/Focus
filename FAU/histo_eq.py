# Implement the histogram equalization in this file
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from otsu import create_greyscale_histogram

img = cv.imread('hello.png',cv.IMREAD_GRAYSCALE)



def cdf(image):
    h,w = image.shape[:2]
    res = create_greyscale_histogram(image)

    p = []
    for i in range(len(res)):
        P = res[i]/(h*w)
        p.append(P)

    C = np.cumsum(p)
    return C


C = cdf(img)
img_new = img.copy()
h,w = img_new.shape[:2]

for i in range(h):
    for j in range(w):
        img_new[i][j] = ((C[img[i][j]] - np.min(C)) / (np.max(C) - np.min(C))) * 255


imgs = np.hstack([img, img_new])
cv.imshow('imgs',imgs)

cv.waitKey(0)
cv.destroyAllWindows()













