import cv2 as cv
import numpy as np


def KN_image_inpainting(image, mask, k=4):
    outputimage = image.copy()
    metalpoints = np.where(mask == 1)
    x = metalpoints[0]
    y = metalpoints[1]
    kernel = np.ones((5, 5))
    dilation = cv.dilate(mask, kernel, iterations=1)
    dilation -= mask
    zeropoints = np.where(dilation == 1)
    targetx = zeropoints[0]
    targety = zeropoints[1]
    number_pixels = len(metalpoints[0])
    for i in range(number_pixels):
        dist = (x[i] - targetx) ** 2 + (y[i] - targety) ** 2
        positions = np.argsort(dist)
        k_points = positions[0:k]
        k_dist = dist[k_points]
        weights = 1 / k_dist / np.sum(1 / k_dist)
        k_tragetx = targetx[k_points]
        k_tragety = targety[k_points]
        k_turple = (k_tragetx, k_tragety)
        k_pixels = image[k_turple]
        outputimage[x[i], y[i]] = np.sum(weights * k_pixels)

    return outputimage
