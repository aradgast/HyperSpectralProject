import numpy as np


def m8(cube):
    """calculate the 8 neighbors average for subtracting the background, include the case when the pixel is on the edge and
    the cube are a 3D or 2D array"""

    # orginal shape of the cube


    rows, cols, bands = cube.shape
    m8cube = np.zeros(shape=(bands, rows, cols), dtype=np.double)
    cube = np.transpose(cube, (2, 0, 1))

    m8cube[:, 1:rows-2, 1:cols-2] = (cube[:, 1:rows-2, 2:cols-1] + cube[:, 1:rows-2, 0:cols-3] + cube[:, 2:rows-1, 1:cols-2] + cube[:, 0:rows-3, 1:cols-2] +
                             cube[:, 2:rows-1, 2:cols-1] + cube[:, 2:rows-1, 0:cols-3] + cube[:, 0:rows-3, 2:cols-1] + cube[:, 0:rows-3, 0:cols-3]) / 8

    # the edge pixels
    row_1 = 0
    row_final = rows
    col_1 = 0
    col_final = cols

    m8cube[:, row_1, 1:cols-2] = (cube[:, row_1, 2:cols-1] + cube[:, row_1, 0:cols-3] + cube[:, row_1+1, 1:cols-2] +
                                 cube[:, row_1+1, 2:cols-1] + cube[:, row_1+1, 0:cols-3]) / 5
    m8cube[:, row_final-1, 1:cols-2] = (cube[:, row_final-1, 2:cols-1] + cube[:, row_final-1, 0:cols-3] +
                                       cube[:, row_final-2, 1:cols-2] + cube[:, row_final-2, 2:cols-1] +
                                       cube[:, row_final-2, 0:cols-3]) / 5
    m8cube[:, 1:rows-2, col_1] = (cube[:, 1:rows-2, col_1+1] + cube[:, 2:rows-1, col_1] + cube[:, 0:rows-3, col_1] +
                                 cube[:, 2:rows-1, col_1+1] + cube[:, 0:rows-3, col_1+1]) / 5
    m8cube[:, 1:rows-2, col_final-1] = (cube[:, 1:rows-2, col_final-2] + cube[:, 2:rows-1, col_final-1] +
                                       cube[:, 0:rows-3, col_final-1] + cube[:, 2:rows-1, col_final-2] +
                                       cube[:, 0:rows-3, col_final-2]) / 5

    # the corner pixels
    m8cube[:, row_1, col_1] = (cube[:, row_1, col_1+1] + cube[:, row_1+1, col_1] + cube[:, row_1+1, col_1+1]) / 3
    m8cube[:, row_1, col_final-1] = (cube[:, row_1, col_final-2] + cube[:, row_1+1, col_final-1] +
                                        cube[:, row_1+1, col_final-2]) / 3
    m8cube[:, row_final-1, col_1] = (cube[:, row_final-1, col_1+1] + cube[:, row_final-2, col_1] +
                                        cube[:, row_final-2, col_1+1]) / 3
    m8cube[:, row_final-1, col_final-1] = (cube[:, row_final-1, col_final-2] + cube[:, row_final-2, col_final-1] +
                                              cube[:, row_final-2, col_final-2]) / 3

    m8cube = np.transpose(m8cube, (1, 2, 0))
    return m8cube


def cov8(cube, m8):
    """calculate the covariance matrix of the cube"""

    rows, cols, bands = cube.shape
    cov = np.zeros(shape=(bands, bands), dtype=np.double)
    for r in range(rows):
        for c in range(cols):
            x1 = (cube[r, c, :] - m8[r, c, :]).reshape((-1, 1))
            cov += np.matmul(x1, np.transpose(x1))
    return cov / (rows * cols)

if __name__ == "__main__":
    a = np.array([[0, 0, 0],
                  [1, 0, 1],
                  [0, 1, 0]])
    b = np.ones(shape=(3, 3, 3))
    b[:, :, 0] = a
    b[:, :, 2] = -a
    print(b)  # print(m8(a))
    print(b.shape)
    print(b.reshape(3, 9))
    print(b)
