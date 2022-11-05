import numpy as np


def m8(cube):
    """calculate the 8,5,3 neighbors average for subtracting the background"""
    is_matrix = False
    if len(cube.shape) == 3:
        rpic, cpic, bands = cube.shape
    elif len(cube.shape) == 2:
        is_matrix = True
        rpic, cpic = cube.shape
        bands = 1
        cube = cube.reshape(rpic, cpic, bands)

    m8cube = np.zeros(shape=(rpic, cpic, bands))
    rsize = rpic - 1
    csize = cpic - 1

    ## 8 neighbors
    m8cube[1:rsize, 1:csize, :] = (cube[0:rsize - 1, 0:csize - 1] + cube[0:rsize - 1, 1:csize] +
                                   cube[0:rsize - 1, 2:csize + 1] + cube[1:rsize, 0:csize - 1] + cube[1:rsize,
                                                                                                 2:csize + 1] +
                                   cube[2:rsize + 1, 0:csize - 1] + cube[2:rsize + 1, 1:csize] + cube[2:rsize + 1,
                                                                                                 2:csize + 1]) / 8

    ## 5 neighbors
    m8cube[0, 1:csize] = (cube[0, 0:csize - 1] + cube[1, 0:csize - 1] + cube[1, 1:csize] + cube[1, 2:csize + 1] +
                          cube[rsize, 2:csize + 1]) / 5
    m8cube[rsize, 1:csize] = (cube[rsize, 0:csize - 1] + cube[rsize - 1, 0:csize - 1] + cube[rsize - 1, 1:csize]
                              + cube[rsize - 1, 2:csize + 1] + cube[rsize, 2:csize + 1]) / 5
    m8cube[1:rsize, 0] = (cube[0:rsize - 1, 0] + cube[0:rsize - 1, 1] + cube[1:rsize, 1] + cube[2:rsize + 1, 1]
                          + cube[2:rsize + 1, 0]) / 5
    m8cube[1:rsize, csize] = (cube[0:rsize - 1, csize] + cube[0:rsize - 1, csize - 1] + cube[1:rsize, csize - 1]
                              + cube[2:rsize + 1, csize - 1] + cube[2:rsize + 1, csize]) / 5

    ## 3 neighbors
    m8cube[0, 0] = (cube[0, 1] + cube[1, 1] + cube[1, 0]) / 3
    m8cube[0, csize] = (cube[0, csize - 1] + cube[1, csize - 1] + cube[1, csize]) / 3
    m8cube[rsize, 0] = (cube[rsize - 1, 0] + cube[rsize - 1, 1] + cube[rsize, 1]) / 3
    m8cube[rsize, csize] = (cube[rsize - 1, csize - 1] + cube[rsize - 1, csize] + cube[rsize, csize - 1]) / 3
    if is_matrix:
        m8cube = m8cube.reshape(rpic, cpic)
    return m8cube


if __name__ == "__main__":
    a = np.array([[0, 0, 0],
                  [1, 0, 1],
                  [0, 1, 0]])
    b = np.ones(shape=(3, 3,3))
    b[:,:,0] = a
    b[:,:,2] = -a
    print(b)  # print(m8(a))
    print(b.shape)
    print(b.reshape(3,9))
    print(b)