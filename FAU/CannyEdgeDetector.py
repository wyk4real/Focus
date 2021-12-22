import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


#
# NO MORE MODULES ALLOwED
#
def gaussFilter(img_in, ksize, sigma):
    """
    fit_lower the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, fit_lowered) kernel and gaussian fit_lowered image (both np.ndarray)
    """
    # TODO

    pad = ksize // 2
    gauss_kernel = np.zeros((ksize, ksize), dtype=np.float64)

    for j in range(-pad, -pad + ksize):
        for i in range(-pad, -pad + ksize):
            gauss_kernel[j + pad, i + pad] = np.exp(-(i ** 2 + j ** 2) / (2 * (sigma ** 2)))

    gauss_kernel = gauss_kernel / (2 * np.pi * (sigma ** 2))
    gauss_kernel = gauss_kernel / np.sum(gauss_kernel)

    result_low = convolve(img_in, gauss_kernel, mode='constant', cval=0.0).astype(np.int64)  #非常重要类型
    return gauss_kernel, result_low


def sobel(img_in):
    """
    applies the sobel fit_lowers to the input image
    watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel fit_lowered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    # TODO

    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gx = convolve(img_in, sx, mode='constant', cval=0.0)
    gy = convolve(img_in, sy, mode='constant', cval=0.0)

    return gx, gy


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel fit_lowered image in x direction (np.ndarray)
    :param gy: sobel fit_lowered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    # TODO

    h, w = gx.shape
    g = np.zeros((h, w))
    theta = np.zeros((h, w))

    for y in range(h):
        for x in range(w):
            g[y, x] = np.sqrt(gx[y, x]**2 + gy[y, x]**2)
            theta[y, x] = np.arctan2(gy[y, x], gx[y, x])

    return g, theta


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    # TODO

    angle = np.rad2deg(angle)
    n = angle//180
    angle = angle - n*180

    if 112.5 <= angle < 157.5:
        angle = 135
    elif 67.5 <= angle < 112.5:
        angle = 90
    elif 22.5 <= angle < 67.5:
        angle = 45
    else:
        angle = 0

    return angle


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    # TODO hint: For 2.3.1 and 2 use the helper method above
    h, w = g.shape
    angle = np.zeros((h, w), dtype=np.int64)
    for y in range(h):
        for x in range(w):
            angle[y, x] = convertAngle(theta[y, x])

    max_sup = np.zeros((h+2, w+2))

    g_new = np.zeros((h + 2, w + 2))
    g_new[1: 1 + h, 1: 1 + w] = g.copy()

    for y in range(h):
        for x in range(w):
            if angle[y, x] == 0:
                if g_new[y+1, x+1] >= g_new[y+1, x] and g_new[y+1, x+1] >= g_new[y+1, x+2]:
                    max_sup[y+1, x+1] = g_new[y+1, x+1]

            elif angle[y, x] == 45:
                if g_new[y+1, x+1] >= g_new[y, x+2] and g_new[y+1, x+1] >= g_new[y+2, x]:
                    max_sup[y+1, x+1] = g_new[y+1, x+1]

            elif angle[y, x] == 90:
                if g_new[y+1, x+1] >= g_new[y, x+1] and g_new[y+1, x+1] >= g_new[y+2, x+1]:
                    max_sup[y+1, x+1] = g_new[y+1, x+1]

            elif angle[y, x] == 135:
                if g_new[y+1, x+1] >= g_new[y, x] and g_new[y+1, x+1] >= g_new[y+2, x+2]:
                    max_sup[y+1, x+1] = g_new[y+1, x+1]

    max_sup = max_sup[1:h+1, 1:w+1]

    return max_sup


def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    # TODO

    h, w = max_sup.shape

    max_sup[max_sup >= t_high] = 255
    max_sup[max_sup <= t_low] = 0

    _max_sup = np.zeros((h + 2, w + 2), dtype=np.float32)
    _max_sup[1: h + 1, 1: w + 1] = max_sup

    nn = np.array(((1, 1, 1), (1, 0, 1), (1, 1, 1)))

    for y in range(1, h + 2):
        for x in range(1, w + 2):
            if _max_sup[y, x] < t_low or _max_sup[y, x] > t_high:
                continue
            if np.max(_max_sup[y - 1:y + 2, x - 1:x + 2] * nn) >= t_high:
                _max_sup[y, x] = 255
            else:
                _max_sup[y, x] = 0

    max_sup = _max_sup[1:h + 1, 1:w + 1]

    return max_sup


def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()

    plt.show()


    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()

    plt.show()

    maxS_img = maxSuppress(g, theta)
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result_low = hysteris(maxS_img, 50, 75)
    plt.imshow(result_low,'gray')
    plt.show()

    return result_low


if __name__ == '__main__':
    img_in = np.array(Image.open('contrast.jpg').convert('L'))
    canny(img_in)



