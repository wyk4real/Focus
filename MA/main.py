import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import cv2 as cv
from skimage.metrics import structural_similarity as ssim


def binarization(image, threshold):
    h, w = image.shape
    new = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            if image[i, j] > threshold:
                new[i, j] = 1
            else:
                new[i, j] = 0
    return new


def get_edge(image):
    kernel = np.ones([3, 3], dtype=np.int32)
    img_new = cv.dilate(image, kernel)
    edge = img_new - image

    return edge


def get_point(image):
    x = []
    y = []
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            if image[i, j] != 0:
                x.append(i)
                y.append(j)
    return x, y


def get_distance(x0, y0, x1, y1):
    x1 = np.array(x1).reshape(1, -1)
    y1 = np.array(y1).reshape(1, -1)

    X = []
    for i in range(len(x0)):
        zwi = x1 - x0[i]
        X.append(zwi)

    X = np.array(X).reshape(len(x0), -1)

    Y = []
    for i in range(len(y0)):
        zwi = y1 - y0[i]
        Y.append(zwi)

    Y = np.array(Y).reshape(len(y0), -1)

    D = X * X + Y * Y
    dist = np.ceil(np.sqrt(D)).astype(np.int32)

    return dist


def pre_processing(dist, n):
    dist = np.sort(dist, axis=1)
    dist = dist[:, :n]

    reciprocal_dist = 1 / dist
    m = 1 / np.sum(reciprocal_dist, axis=1)

    weights = np.zeros((dist.shape[0], n), dtype=np.float32)

    for i in range(len(m)):
        weights[i, :] = reciprocal_dist[i, :] * m[i]

    return dist, weights


def get_index(distance, dist):
    INDEX = []
    h, w = dist.shape
    for i in range(h):
        zwi = []
        for j in range(w):
            d = dist[i, j]
            index = list(np.where(distance[i, :] == d)[0])
            for l in range(len(index)):
                if len(zwi) < w and index[l] not in zwi:
                    zwi.append(index[l])

        INDEX.append(zwi)
    INDEX = np.array(INDEX, dtype=np.int32).reshape(h, w)

    return INDEX


def get_pixel_value(index, x1, y1, image):
    h, w = index.shape
    pixel_value = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            x = x1[index[i, j]]
            y = y1[index[i, j]]
            pixel_value[i, j] = image[x, y]

    return pixel_value


def nn_interpolation(image, mask, number_of_neighbours):
    mask = binarization(mask, 0.2)  # 金属图像二值化，取阈值为0.2
    inverse_mask = 1 - mask  # 金属图像取反
    image = image * inverse_mask  # 生成待修复的图像
    edge = get_edge(mask)  # 获取边界

    x1, y1 = get_point(edge)  # 获取所有边界点的坐标(4665)
    x0, y0 = get_point(mask)  # 获取所有待修复点的坐标(71940)

    distance = get_distance(x0, y0, x1, y1)  # 计算修复点到边界点的欧式距离(71940*4665)

    dist, weights = pre_processing(distance, number_of_neighbours)  # 设定关联点的个数，以及获得关联点的权重(71940*3)

    index = get_index(distance, dist)  # 获取最邻近点坐标的索引(71940*3)

    pixel_value = get_pixel_value(index, x1, y1, image)  # 生成前n个点的像素值矩阵（71940*3）

    value = np.sum(pixel_value * weights, axis=1)  # 像素值矩阵点乘权重矩阵然后求和得到最终的像素值

    for q in range(len(value)):
        image[x0[q], y0[q]] = value[q]

    return image


def mse(imageA, imageB, mask, threshold):
    D = []
    h, w = mask.shape
    bi_mask = binarization(mask, threshold)
    for i in range(h):
        for j in range(w):
            if bi_mask[i, j] != 0:
                D.append(np.square(imageA[i, j] - imageB[i, j]))

    MSE = np.sum(D) / len(D)

    return MSE


def compare_images(imageA, imageB):
    m = mse(imageA, imageB, m1_1, 0.2)
    s = ssim(imageA, imageB)

    fig = plt.figure('Orinianl Image vs Inpainting Image')
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")
    fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    plt.show()


if __name__ == '__main__':
    ms1 = tiff.imread('../dataset/Projection_M/projection1.tif')
    vs1 = tiff.imread('../dataset/Projection_V/projection1.tif')
    vms1 = tiff.imread('../dataset/Projection_V_M/projection1.tif')

    m1_1 = ms1[0]
    v1_1 = vs1[0]
    vm1_1 = vms1[0]

    img_new = nn_interpolation(vm1_1, m1_1, 3)
    plt.imshow(img_new, cmap=plt.cm.gray)
    plt.show()

    compare_images(v1_1, img_new)







