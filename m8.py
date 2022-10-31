import numpy as np


def m8(cube):
    rpic, cpic, bands = cube.shape
    m8cube = np.zeros(shape=(rpic, cpic, bands))
    rsize = rpic - 1
    csize = cpic - 1
    col = np.array([i for i in range(1, csize)])
    col_p1 = np.array([i + 1 for i in range(1, csize)])
    col_m1 = np.array([i - 1 for i in range(1, csize)])
    row = np.array([i for i in range(1, rsize)])
    row_p1 = np.array([i + 1 for i in range(1, rsize)])
    row_m1 = np.array([i - 1 for i in range(1, rsize)])
    # print(cube[0:rsize-1, 0:csize-1].shape)
    # print(cube[0:rsize-1, 1:csize].shape)
    # print(cube[0:rsize-1, 2:csize+1].shape)
    # print(cube[1:rsize, 0:csize-1].shape)
    # print(cube[1:rsize, 2:csize+1].shape)
    # print(cube[2:rsize+1, 2:csize+1].shape)
    # print(cube[2:rsize+1, 0:csize].shape)
    # print(cube[2:rsize+1, 0:csize-1].shape)
    ## 8 neighbors
    m8cube[1:rsize, 1:csize, :] = (cube[0:rsize-1, 0:csize-1] + cube[0:rsize-1, 1:csize] +
                           cube[0:rsize-1, 2:csize+1] + cube[1:rsize, 0:csize-1] + cube[1:rsize, 2:csize+1] +
                           cube[2:rsize+1, 0:csize-1] + cube[2:rsize+1, 1:csize] + cube[2:rsize+1, 2:csize+1]) / 8
    ## TODO
    m8cube[:, 0, col] = (cube[:, 0, col_m1] + cube[:, 1, col_m1] + cube[:, 1, col] + cube[:, 1, col_p1] + cube[:, rsize,
                                                                                                          col_p1]) / 5
    m8cube[:, rsize, col] = (cube[:, rsize, col_m1] + cube[:, rsize - 1, col_m1] + cube[:, rsize - 1, col]
                             + cube[:, rsize - 1, col_p1] + cube[:, rsize, col_p1]) / 5
    m8cube[:, row, 0] = (cube[:, row_m1, 0] + cube[:, row_m1, 1] + cube[:, row, 1] + cube[:, row_p1, 1]
                         + cube[:, row_p1, 0]) / 5
    m8cube[:, row, csize] = (cube[:, row_m1, csize] + cube[:, row_m1, csize - 1] + cube[:, row, csize - 1]
                             + cube[:, row_p1, csize - 1] + cube[:, row_p1, csize]) / 5

    m8cube[:, 0, 0] = (cube[:, 0, 1] + cube[:, 1, 1] + cube[:, 1, 0]) / 3
    m8cube[:, 0, csize] = (cube[:, 0, csize - 1] + cube[:, 1, csize - 1] + cube[:, 1, csize]) / 3
    m8cube[:, rsize, 0] = (cube[:, rsize - 1, 0] + cube[:, rsize - 1, 1] + cube[:, rsize, 1]) / 3
    m8cube[:, rsize, csize] = (cube[:, rsize - 1, csize - 1] + cube[:, rsize - 1, csize] + cube[:, rsize,
                                                                                           csize - 1]) / 3

    return m8cube


if __name__ == "__main__":
    a = np.array([[0, 0, 0],
                  [1, 0, 1],
                  [0, 1, 0]])
    b = np.ones(shape=(10, 3, 3))
    b[0] = a
    b[2] = -a
    print(m8(b))  # print(m8(a))
