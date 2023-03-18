import numpy as np
from scipy.ndimage import generic_filter
from find_nu import find_nu
from m8 import m8
from spectral import *
import matplotlib.pyplot as plt
from pca import pca
from cov_m8 import cov_m8

from local_meanNcovariance import local_mean_covariance


class ArtificialHyperspectralCube:
    """ x refers to the original data
    y refers to the transformed data(PCA)"""

    def __init__(self, header):
        self.data = open_image(header)
        self.cube = self.data.load()
        self.rows, self.cols, self.bands = self.data.shape

        self.x_mean, self.x_cov = local_mean_covariance(self.cube)

        self.y = pca(self.cube, self.x_mean, self.x_cov)

        self.y_mean, self.y_cov = local_mean_covariance(self.y)

        self.nu_x = find_nu(self.x_np, self.m8x, self.cov_x, False)
        self.nu_y = find_nu(self.y_np, self.m8y, self.cov_y, False)

        # Z cube ############
        self._create_z_cube(self.nu_y)
        self.cov = cov_m8(self.data)
        self.m8 = np.array(m8(self.data))
        self.data -= self.m8
        self.nu = find_nu(self.data, self.m8, self.cov, False)

    def _create_z_cube(self, nu_vec):
        self.data = np.zeros(shape=(self.rowNum, self.colNum, self.bandsNum))
        for band in range(self.bandsNum):
            self.data[:, :, band] += np.random.standard_t(nu_vec[band], size=(self.rowNum, self.colNum))
            self.data[:, :, band] /= self.data[:, :, band].std()

        self.data = np.array(self.data)


if __name__ == "__main__":
    z = ArtificialHyperspectralCube('D1_F12_H2_Cropped_des_Aligned.hdr')
    plt.plot([i for i in range(len(z.nu))], z.nu)
    plt.plot([i for i in range(len(z.m8))], z.m8)
    plt.show()

    pass
