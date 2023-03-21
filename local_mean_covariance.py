############################################################################################################
# All right reserved by BGU, 2023
# Author: Arad Gast
# Date: 03/2023
# Description: This code is for calculating the local mean and covariance matrix of a hyperspectral image
#              The code is based on the course of "Hyperspectral Image Processing" by Prof. Stanly Rotman

#############################################################################################################
import numpy as np


def m8(cube):
    """this function calculate the 8 neighbors average for subtracting the background,
     include the case when the pixel is on the edge and
    the cube are a 3D
    :param cube: the cube of the image
    :return: the 8 neighbors average cube in this shape"""

    row_num, col_num, band_num = cube.shape
    m8cube = np.zeros(shape=(row_num, col_num, band_num), dtype=np.double)

    m8cube[1:row_num-1, 1:col_num-1] = (cube[1:row_num-1, 2:col_num] + cube[1:row_num-1, 0:col_num-2] +
                                        cube[2:row_num, 1:col_num-1] + cube[0:row_num-2, 1:col_num-1] +
                                        cube[2:row_num, 2:col_num] + cube[2:row_num, 0:col_num-2] +
                                        cube[0:row_num-2, 2:col_num] + cube[0:row_num-2, 0:col_num-2]) / 8

    # the edge pixels
    m8cube[0, 1:col_num-1] = np.squeeze((cube[0, 2:col_num] + cube[0, 0:col_num-2] +
                              cube[1, 1:col_num-1] + cube[1, 2:col_num] + cube[1, 0:col_num-2]) / 5)
    m8cube[row_num-1, 1:col_num-1] = np.squeeze((cube[row_num-1, 2:col_num] + cube[row_num-1, 0:col_num-2] +
                                      cube[row_num-2, 0:col_num-2] + cube[row_num-2, 1:col_num-1] + cube[row_num-2, 2:col_num]) / 5)

    m8cube[1:row_num-1, 0] = np.squeeze((cube[0:row_num-2, 0] + cube[2:row_num, 0] +
                              cube[0:row_num-2, 1] + cube[2:row_num, 1] + cube[1:row_num-1, 1]) / 5)
    m8cube[1:row_num-1, col_num-1] = np.squeeze((cube[0:row_num-2, col_num - 1] + cube[2:row_num, col_num-1] +
                                      cube[0:row_num-2, col_num - 2] + cube[1:row_num-1, col_num - 2] + cube[2:row_num, col_num - 2]) / 5)

    # the corner pixels
    m8cube[0, 0] = np.squeeze((cube[0, 1] + cube[1, 0] + cube[1, 1]) / 3)
    m8cube[0, col_num-1] = np.squeeze((cube[0, col_num-2] + cube[1, col_num-1] + cube[1, col_num-2]) / 3)
    m8cube[row_num-1, 0] = np.squeeze((cube[row_num-1, 1] + cube[row_num-2, 0] + cube[row_num-2, 1]) / 3)
    m8cube[row_num-1, col_num-1] = np.squeeze((cube[row_num-1, col_num-2] + cube[row_num-2, col_num-1] +
                                               cube[row_num-2, col_num-2]) / 3)

    return m8cube


def cov8(cube, m8):
    """this function calculate the covariance matrix of the cube using the 8 neighbors average
    :param cube: the cube of the image
    :param m8: the 8 neighbors average cube
    :return: the covariance matrix of the cube"""

    rows, cols, bands = cube.shape
    cov = np.zeros(shape=(bands, bands))
    for r in range(rows):
        for c in range(cols):
            x1 = (cube[r, c, :] - m8[r, c, :]).reshape((-1, 1))
            cov += np.matmul(x1, np.transpose(x1))
    return cov / (rows * cols)

if __name__ == "__main__":
    import spectral as spy
    cube = spy.open_image('D1_F12_H1_Cropped.hdr')
    a = cube.load(dtype='double')
    m = m8(a)
    c = cov8(a, m)
    print(cov8(a, m).shape)
