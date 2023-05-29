############################################################################################################
# All right reserved by BGU, 2023
# Author: Arad Gast, Ido Levokovich
# Date: 03/2023
# Description: This file contains the PCA function

#############################################################################################################

import numpy as np
from local_mean_covariance import get_m8, get_cov8
from legends import *

def get_pca(data, mean=None, cov=None):
    """This function calculate the PCA of the data cube to create un-correlated bands
    param data: the data cube
    param mean: the mean of the data cube
    param cov: the covariance of the data cube
    return: the PCA of the data cube"""

    # get the shape of the cube
    row, col, bands = data.shape
    cube = np.zeros(shape=(row, col, bands), dtype=PRECISION)
    if mean is None:
        mean = get_m8(data)
    if cov is None:
        cov = get_cov8(data, mean)
    data -= mean

    eigval, eigvec = np.linalg.eig(cov)

    scale_eigvec = np.matmul(np.linalg.inv(np.diag(np.sqrt(eigval))), eigvec.T, dtype=PRECISION)

    # project the data
    for r in range(row):
        for c in range(col):
            cube[r, c, :] = np.matmul(scale_eigvec, data[r, c, :], dtype=PRECISION)

    return cube, eigvec


if __name__ == "__main__":
    import spectral as spy
    import matplotlib.pyplot as plt

    # load the data
    data = spy.open_image('D1_F12_H2_Cropped_des_Aligned.hdr')
    # convert the data to a numpy array
    data = np.array(data.open_memmap())
    # perform PCA
    cube, cov, cov2 = get_pca(data)
    # plot the data
    plt.imshow(cube[:, :, 0])
    plt.show()
    plt.imshow(cov)
    plt.show()
    plt.imshow(cov2)
    plt.colorbar()
    plt.show()

    pass
