from PIL import Image
import numpy as np
from scipy.signal import convolve
from scipy.signal import gaussian


def make_gauss_kernel(ksize, sigma):
    G = np.zeros((ksize, ksize))  # Init of Matrix

    para1 = 1 / (2 * np.pi * np.square(sigma))  # Parameter's Init
    para2 = 2 * np.square(sigma)
    n = int((ksize - 1) / 2)  # ksize = 5 ==> n = 2

    for i in range(0, n + 1):  # kernel matrix G(x,y)
        for j in range(0, n + 1):
            G[i, j] = para1 * np.exp(-(np.square(n - i) + np.square(n - j)) / para2)
            G[n + (n - i), n + (n - j)] = G[i, j]
            G[n + (n - i), j] = G[i, j]
            G[i, n + (n - j)] = G[i, j]
    G[n, n] = para1

    sum_G = np.sum(G)  # normalize the matrix such that all elements sum up to 1.
    G = G / sum_G

    return G  # implement the Gaussian kernel here


def slow_convolve(arr, k):
    # Initiate a new image with the same size as the input image:
    arr_h, arr_w = arr.shape
    img_new = np.zeros((arr_h, arr_w))
    sum_index = 0

    # Zero - Padding:
    h, w = k.shape  # h => U; w => V
    k_sum = np.zeros((h, w))
    down_h = np.int(np.floor(h / 2))  # rounding h down.
    down_w = np.int(np.floor(w / 2))  # rounding w down.
    up_h = np.int(np.ceil(h / 2))  # rounding h up.
    up_w = np.int(np.ceil(w / 2))  # rounding w up.)

    padding_array = np.zeros([arr_h + 2 * down_h, arr_w + 2 * down_w])
    padding_array[down_h:arr_h + down_h, down_w:arr_w + down_w] = arr

    for i in range(0, arr_h):  # Go over each pixel
        for j in range(0, arr_w):
            for u in range(-down_h, up_h):  # calculate the value for this pixel using the equation 1
                for v in range(-down_w, up_w):
                    k_sum[u + down_h, v + down_w] = k[u + down_h, v + down_w] * padding_array[
                        i + h - 1 - (u + down_h), j + w - 1 - (v + down_w)]
            img_new[i, j] = k_sum.sum()

    return img_new  # implement the convolution with padding here


def clip(array, minimum, maximum):
    result = np.where(array < minimum, minimum, array)
    result = np.where(result > maximum, maximum, result)
    return result
    # clip the values to the range [0,255] (warme-up exercise)


if __name__ == '__main__':
    k1 = make_gauss_kernel(3, 10)  # todo: find better parameters
    k2 = make_gauss_kernel(5, 50)
    k3 = make_gauss_kernel(10, 100)

    # TODO: chose the image you prefer:
    im1 = np.array(Image.open('input1.jpg'))
    im2 = np.array(Image.open('input2.jpg'))
    im3 = np.array(Image.open('input3.jpg'))

    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result

    # im1 bluring:
    arr0 = im1[:, :, 0]
    arr1 = im1[:, :, 1]
    arr2 = im1[:, :, 2]
    res0 = slow_convolve(arr0, k1)
    res1 = slow_convolve(arr1, k1)
    res2 = slow_convolve(arr2, k1)
    arr0 = arr0 + (arr0 - res0)
    im1[:, :, 0] = clip(arr0, 0, 255)
    arr1 = arr1 + (arr1 - res1)
    im1[:, :, 1] = clip(arr1, 0, 255)
    arr2 = arr2 + (arr2 - res2)
    im1[:, :, 2] = clip(arr2, 0, 255)
    new_im = Image.fromarray(im1)  # data二维图片矩阵。
    new_im.show()
    new_im.save('input1_new.jpg')

    # im2 bluring:
    arr0 = im2[:, :, 0]
    arr1 = im2[:, :, 1]
    arr2 = im2[:, :, 2]
    res0 = slow_convolve(arr0, k2)
    res1 = slow_convolve(arr1, k2)
    res2 = slow_convolve(arr2, k2)
    arr0 = arr0 + (arr0 - res0)
    im2[:, :, 0] = clip(arr0, 0, 255)
    arr1 = arr1 + (arr1 - res1)
    im2[:, :, 1] = clip(arr1, 0, 255)
    arr2 = arr2 + (arr2 - res2)
    im2[:, :, 2] = clip(arr2, 0, 255)
    new_im = Image.fromarray(im2)  # data二维图片矩阵。
    new_im.show()
    new_im.save('input2_new.jpg')

    # im3 bluring:
    arr0 = im3[:, :, 0]
    arr1 = im3[:, :, 1]
    arr2 = im3[:, :, 2]
    res0 = slow_convolve(arr0, k3)
    res1 = slow_convolve(arr1, k3)
    res2 = slow_convolve(arr2, k3)
    arr0 = arr0 + (arr0 - res0)
    im3[:, :, 0] = clip(arr0, 0, 255)
    arr1 = arr1 + (arr1 - res1)
    im3[:, :, 1] = clip(arr1, 0, 255)
    arr2 = arr2 + (arr2 - res2)
    im3[:, :, 2] = clip(arr2, 0, 255)
    new_im = Image.fromarray(im3)  # data二维图片矩阵。
    new_im.show()
    new_im.save('input3_new.jpg')




