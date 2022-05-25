import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff


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


def edge_detection(image):
    h, w = image.shape
    edge = np.zeros((h, w), dtype=np.float32)
    for i in range(w):
        for j in range(1, h-1):
            if image[j, i] != 0:
                if (image[j - 1, i] == 0 and image[j + 1, i] != 0) or (image[j - 1, i] != 0 and image[j + 1, i] == 0):
                    edge[j, i] = 1
    return edge


def edge_points(image):
    points = []
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            if image[i, j] == 1:
                coordinate = (i, j)
                points.append(coordinate)
    return points


def get_distance(image, point):
    h, w = image.shape
    dist = []
    for n in range(len(point)):
        d = []
        x = point[n][0]
        y = point[n][1]

        for i in range(h):
            if image[i, y] != 0:
                dist_x = np.abs(x - i)
                d.append(dist_x)

        for j in range(w):
            if image[x, j] != 0:
                dist_y = np.abs(y - j)
                d.append(dist_y)

        dist.append(np.min(d))
        d = []
    return dist


def gauss_weights(size, sigma):
    pad = size // 2
    gauss_kernel = np.zeros((size, size), dtype=np.float32)

    for j in range(-pad, -pad + size):
        for i in range(-pad, -pad + size):
            gauss_kernel[j + pad, i + pad] = np.exp(-(i ** 2 + j ** 2) / (2 * (sigma ** 2)))

    gauss_kernel = gauss_kernel / (2 * np.pi * (sigma ** 2))
    gauss_kernel = gauss_kernel / np.sum(gauss_kernel)

    return gauss_kernel


if __name__ == '__main__':
    # 步骤一： CT图读取
    ms1 = tiff.imread('./dataset/Projection_M/projection1.tif')
    vs1 = tiff.imread('./dataset/Projection_V/projection1.tif')
    vms1 = tiff.imread('./dataset/Projection_V_M/projection1.tif')

    m1_1 = ms1[0]
    v1_1 = vs1[0]
    vm1_1 = vms1[0]

    # 步骤二： 金属植入物图像二值化，取反以及生成待修复得CT图像
    binary_m1_1 = binarization(m1_1, 0.2)
    inverse_m1_1 = 1 - binary_m1_1
    re_vm1_1 = vm1_1 * inverse_m1_1

    # 步骤三： 获取待修复点的坐标
    points = edge_points(binary_m1_1)

    # 步骤四： 获取待修复点离边界点的最短距离
    distance = get_distance(re_vm1_1, points)

    # 步骤五： 对于每个修复点生成高斯权重核，并进行插值
    for i in range(len(distance)):
        d = distance[i]
        weight = gauss_weights(2 * d - 1, 1)

        relation = np.zeros((2 * d - 1, 2 * d - 1), dtype=np.float32)

        point = points[i]
        re_vm1_1[point] = weight * relation

    plt.imshow(re_vm1_1, plt.cm.gray)
    plt.show()

    # # 可视化

    # # 对待修复的图片和金属图片进行拉东变换并保存在当前文件夹下
    # fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    # ax[0][0].set_title("image to be repaired")
    # ax[0][0].imshow(re_img, cmap=plt.cm.gray)
    #
    # ax[0][1].set_title("metal")
    # ax[0][1].imshow(metal, cmap=plt.cm.gray)
    #
    # theta = np.linspace(0., 180., num=re_img.shape[0], endpoint=False)
    #
    # sinogram_re_img = radon(re_img, theta=theta, circle=True)
    # cv.imwrite('sinogram_re_img.jpg', sinogram_re_img)
    # ax[1][0].set_title("Radon transform\n(re_img_Sinogram)")
    # ax[1][0].set_xlabel("Projection angle (deg)")
    # ax[1][0].set_ylabel("Projection position (pixels)")
    # ax[1][0].imshow(sinogram_re_img, cmap=plt.cm.gray,
    #                 extent=(0, 180, 0, sinogram_re_img.shape[0]), aspect='auto')
    #
    # sinogram_mt = radon(metal, theta=theta, circle=True)
    # cv.imwrite('sinogram_mt.jpg', sinogram_mt)
    # ax[1][1].set_title("Radon transform\n(metal_Sinogram)")
    # ax[1][1].set_xlabel("Projection angle (deg)")
    # ax[1][1].set_ylabel("Projection position (pixels)")
    # ax[1][1].imshow(sinogram_mt, cmap=plt.cm.gray,
    #                 extent=(0, 180, 0, sinogram_mt.shape[0]), aspect='auto')
    #
    # fig.tight_layout()
    # plt.savefig('./result.jpg')
    # plt.show()











































