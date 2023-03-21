import numpy as np
from scipy.ndimage import generic_filter
from find_nu import find_nu
from local_mean_covariance import m8, cov8
from spectral import *
import matplotlib.pyplot as plt
from pca import pca



class ArtificialHyperspectralCube:
    """ this class initalize an artifcial hyperspectral cube according to the original data that was given by the
     header file.

     the class contains the following attributes:
     data - the original data
     cube - the original data in a 3D array
     rows - the number of rows in the original data
     cols - the number of columns in the original data
     bands - the number of bands in the original data
     x_mean - the local mean of the original data
     x_cov - the local covariance of the original data
     y - the PCA of the original data
     y_mean - the local mean of the PCA
     y_cov - the local covariance of the PCA
     nu_x - the estimation of df(degree of freedom) for the original data
     nu_y - the estimation of df(degree of freedom) for the PCA dataset
     z - the artificial hyperspectral cube
     cov - the local covariance of the artificial hyperspectral cube
     m8 - the local mean of the artificial hyperspectral cube
     nu - the estimation of df(degree of freedom) for the artificial hyperspectral cube

     """

    def __init__(self, header):
        self.data = open_image(header)
        self.cube = self.data.load()
        self.rows, self.cols, self.bands = self.data.shape

        self.x_mean = m8(self.cube)
        self.x_cov = cov8(self.cube, self.x_mean)

        self.y = pca(self.cube, self.x_mean, self.x_cov)

        self.y_mean = m8(self.y)
        self.y_cov = cov8(self.y, self.y_mean)

        # self.nu_x = find_nu(self.x_np, self.m8x, self.cov_x, False)
        # self.nu_y = find_nu(self.y_np, self.m8y, self.cov_y, False)

        # Z cube ############
        self.__create_z_cube([2 for i in range(self.bands)])
        self.m8 = m8(self.artificial_data)
        self.cov = cov8(self.artificial_data, self.m8)
        # self.nu = find_nu(self.data, self.m8, self.cov, False)

    def __create_z_cube(self, nu_vec):
        self.artificial_data = np.zeros(shape=(self.rows, self.cols, self.bands))
        for band in range(self.bands):
            self.artificial_data[:, :, band] += np.random.standard_t(nu_vec[band], size=(self.rows, self.cols))
            self.artificial_data[:, :, band] /= self.artificial_data[:, :, band].std()

        self.artificial_data = np.array(self.artificial_data)


if __name__ == "__main__":
    z = ArtificialHyperspectralCube('D1_F12_H2_Cropped_des_Aligned.hdr')
    plt.plot([i for i in range(len(z.nu))], z.nu)
    plt.plot([i for i in range(len(z.m8))], z.m8)
    plt.show()

    pass
