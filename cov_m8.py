import numpy as np

def cov_m8(cube, m8):

    N = cube.shape[0] * cube.shape[1]
    meaned_cube = cube - m8
    phi = np.zeros((cube.shape[2], cube.shape[2]))
    for r in range(cube.shape[0]):
        for c in range(cube.shape[1]):
            x1 = np.reshape(meaned_cube[r, c, :], [meaned_cube.shape[2], 1])
            phi += np.matmul(x1, np.transpose(x1))
    return phi / N


    # cube -= m8(cube)
    # cube = np.reshape(cube, (cube.shape[0]*cube.shape[1], cube.shape[2]))
    # return np.matmul(np.transpose(cube), cube)


if __name__ == "__main__":

    a = np.random.rand(10, 15, 5)
    print(cov_m8(a).shape)


