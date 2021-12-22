import cv2
import mpmath
import numpy as np
from matplotlib import pyplot as plt
#
# NO OTHER IMPORTS ALLOWED
#


def create_greyscale_histogram(img):
    '''
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''

    # hist = cv2.calcHist(img, [0], None, [256], [0, 256])
    # plt.plot(hist)
    # plt.show()


    img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]

    y = np.zeros(256,dtype=np.float64)
    for i in range(h):
        for j in range(w):
            pixel = img[i, j]
            y[pixel] = y[pixel] + 1

    return y


def binarize_threshold(img, t):
    '''
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    # ret, binary = cv.threshold(img, t, 255, cv.THRESH_BINARY)
    # cv.imshow('binary', binary)
    # cv.waitKey(0)
    # return binary

    h, w = img.shape[:2]

    for i in range(h):
        for j in range(w):
            pixel = img[i,j]
            if pixel < t:
                img[i,j] = 0
            else:
                img[i,j] = 255

    return img


def p_helper(hist, theta: int):
    '''
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    '''

    N = np.sum(hist)
    Z = 0
    for i in range(theta+1):
        Z += hist[i]

    p0 = Z / N
    p1 = 1 - p0

    return p0, p1


def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and m1
    :param hist: histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1
    '''


    s_mu0_z = p0 * np.sum(hist)
    s_mu0_w = 0
    for i in range(theta):
        s_mu0_w += i * hist[i]

    mu0 = s_mu0_w / s_mu0_z if s_mu0_z > 0 else 0.

    s_mu1_z = p1 * np.sum(hist)
    s_mu1_w = 0
    for i in range(theta, 256):
        s_mu1_w += i * hist[i]

    mu1 = s_mu1_w / s_mu1_z if s_mu1_z > 0 else 0.

    return mu0, mu1


def calculate_otsu_threshold(hist):
    '''
    calculates theta according to otsus method

    :param hist: 1D array
    :return: threshold (int)
    '''
    # TODO initialize all needed variables

    # TODO change the histogram, so that it visualizes the probability distribution of the pixels
    # --> sum(hist) = 1

    # TODO loop through all possible thetas

        # TODO compute p0 and p1 using the helper function

        # TODO compute mu and m1 using the helper function

        # TODO compute variance

        # TODO update the threshold



    sum = np.sum(hist)
    for i in range(len(hist)):
        hist[i] = hist[i] / sum

    theta = np.linspace(0,255,256)

    plt.bar(theta,hist)
    plt.show()

    G = []

    for t in range(len(theta)):
        p0,p1 = p_helper(hist,t)
        mu0,mu1 = mu_helper(hist,t,p0,p1)

        g = p0 * p1 * (mu0 - mu1) ** 2
        G.append(g)

    T = np.argmax(G)
    return T


def otsu(img):
    '''
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    '''
    # TODO


    hist = create_greyscale_histogram(img)
    t = calculate_otsu_threshold(hist)
    print(t)
    binary = binarize_threshold(img,t)

    return binary







