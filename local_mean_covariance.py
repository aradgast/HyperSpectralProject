############################################################################################################
# All right reserved by BGU, 2023
# Author: Arad Gast, Ido Levokovich
# Date: 03/2023
# Description: This code is for calculating the local mean and covariance matrix of a hyperspectral image
#              The code is based on the course of "Hyperspectral Image Processing" by Prof. Stanly Rotman

#############################################################################################################
import numpy as np

PRECISION = np.double


def get_m8(cube, method='local'):
    """this function calculate the 8 neighbors average for subtracting the background,
     include the case when the pixel is on the edge and
    the cube are a 3D
    :param cube: the cube of the image
    :return: the 8 neighbors average cube in this shape"""

    row_num, col_num, band_num = cube.shape
    if method == 'local':
        m8cube = np.zeros(shape=(row_num, col_num, band_num), dtype=PRECISION)

        m8cube[1:row_num - 1, 1:col_num - 1] = (cube[1:row_num - 1, 2:col_num] + cube[1:row_num - 1, 0:col_num - 2] +
                                                cube[2:row_num, 1:col_num - 1] + cube[0:row_num - 2, 1:col_num - 1] +
                                                cube[2:row_num, 2:col_num] + cube[2:row_num, 0:col_num - 2] +
                                                cube[0:row_num - 2, 2:col_num] + cube[0:row_num - 2, 0:col_num - 2]) / 8

        # the edge pixels
        m8cube[0, 1:col_num - 1] = np.squeeze((cube[0, 2:col_num] + cube[0, 0:col_num - 2] +
                                               cube[1, 1:col_num - 1] + cube[1, 2:col_num] + cube[1, 0:col_num - 2]) / 5)
        m8cube[row_num - 1, 1:col_num - 1] = np.squeeze((cube[row_num - 1, 2:col_num] + cube[row_num - 1, 0:col_num - 2] +
                                                         cube[row_num - 2, 0:col_num - 2] + cube[row_num - 2,
                                                                                            1:col_num - 1] + cube[
                                                                                                             row_num - 2,
                                                                                                             2:col_num]) / 5)

        m8cube[1:row_num - 1, 0] = np.squeeze((cube[0:row_num - 2, 0] + cube[2:row_num, 0] +
                                               cube[0:row_num - 2, 1] + cube[2:row_num, 1] + cube[1:row_num - 1, 1]) / 5)
        m8cube[1:row_num - 1, col_num - 1] = np.squeeze((cube[0:row_num - 2, col_num - 1] + cube[2:row_num, col_num - 1] +
                                                         cube[0:row_num - 2, col_num - 2] + cube[1:row_num - 1,
                                                                                            col_num - 2] + cube[2:row_num,
                                                                                                           col_num - 2]) / 5)

        # the corner pixels
        m8cube[0, 0] = np.squeeze((cube[0, 1] + cube[1, 0] + cube[1, 1]) / 3)
        m8cube[0, col_num - 1] = np.squeeze((cube[0, col_num - 2] + cube[1, col_num - 1] + cube[1, col_num - 2]) / 3)
        m8cube[row_num - 1, 0] = np.squeeze((cube[row_num - 1, 1] + cube[row_num - 2, 0] + cube[row_num - 2, 1]) / 3)
        m8cube[row_num - 1, col_num - 1] = np.squeeze((cube[row_num - 1, col_num - 2] + cube[row_num - 2, col_num - 1] +
                                                       cube[row_num - 2, col_num - 2]) / 3)

    elif method == 'global':
        m8cube = np.mean(cube, (0, 1))

    else:
        raise ValueError('method must be "local" or "global"')

    return m8cube


def get_cov8(cube, m8_cube = None, method='local'):
    """this function calculate the covariance matrix of the cube using the 8 neighbors average
    :param cube: the cube of the image
    :param m8: the 8 neighbors average cube
    :return: the covariance matrix of the cube"""

    if m8_cube is None:
        m8_cube = get_m8(cube, method)
    rows, cols, bands = cube.shape
    x = cube - m8_cube  # subtract mean
    x = x.reshape(rows * cols, bands)  # flatten to 2D array
    cov = np.cov(x, rowvar=False, bias=True)  # compute covariance
    return cov


if __name__ == "__main__":
    import spectral as spy

    cube = spy.open_image('data/D1_F12_H1_Cropped.hdr')
    a = cube.load(dtype='double')
    m = get_m8(a)
    c = get_cov8(a, m)
    print(get_cov8(a, m).shape)
