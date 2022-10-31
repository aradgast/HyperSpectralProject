import numpy as np


def find_nu(cube, mean_matrix, cov):
    bands = cube.shape[2]
    nu = np.zeros((bands, 1))
    for i in range(bands):
        r = np.abs(cube[:, :, i] / np.sqrt(cov[i, i]))
        k = np.mean(np.power(r, 3)) / np.mean(r)
        nu[i] = 2 + k / (k - 2)
    return nu
