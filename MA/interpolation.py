# 对单通道图片使用线性插值
import numpy as np
import cv2 as cv


def nn_interpolation(image, new_h, new_w):
    h, w = image.shape
    if h == new_h and w == new_w:
        return image.copy()
    new_img = np.zeros((new_h, new_w), dtype=np.uint8)
    scale_x, scale_y = w / new_w, h / new_h

    for i in range(new_h):
        for j in range(new_w):
            row = i * scale_y
            col = j * scale_x
            near_row = round(row)
            near_col = round(col)
            if near_row == h or near_col == w:
                near_row -= 1
                near_col -= 1

            new_img[i][j] = image[near_row][near_col]

    return new_img


def bilinear_interpolation(image, new_h, new_w):
    h, w = image.shape
    if h == new_h and w == new_w:
        return image.copy()
    new_img = np.zeros((new_h, new_w), dtype=np.uint8)
    scale_x, scale_y = w / new_w, h / new_h

    for i in range(new_h):
        for j in range(new_w):
            row = i * scale_y
            col = j * scale_x
            row_int = int(row)
            col_int = int(col)

            u = row - row_int
            v = col - col_int

            if row_int == h - 1 or col_int == w - 1:
                row_int -= 1
                col_int -= 1

            new_img[i][j] = (1 - u) * (1 - v) * image[row_int][col_int] + \
                            (1 - u) * v * image[row_int][col_int + 1] + \
                            u * (1 - v) * image[row_int + 1][col_int] + \
                            u * v * image[row_int + 1][col_int + 1]
#  f(i+u, j+v) = (1-u) * (1-v) * f(i, j) + (1-u) * v * f(i, j+1) + u * (1-v) * f(i+1, j) + u * v * f(i+1, j+1)
    return new_img


if __name__ == '__main__':
    img1 = cv.imread('no.1.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('no.2.jpg', cv.IMREAD_GRAYSCALE)

    # img_nn = nn_interpolation(img, 1080, 1920)
    img_bi1 = bilinear_interpolation(img1, 1080, 1920)
    img_bi2 = bilinear_interpolation(img2, 1080, 1920)
    cv.imwrite('no.1_new.jpg', img_bi1)
    cv.imwrite('no.2_new.jpg', img_bi2)










