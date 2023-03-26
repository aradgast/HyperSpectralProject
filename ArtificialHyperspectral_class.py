########################################################################################################################
# All right reserved by BGU, 2023
# Author: Arad Gast, Ido Levokovich
# Date: 03/2023
# Description: this file contains the ArtificialHyperspectralCube class

########################################################################################################################

import numpy as np
from local_mean_covariance import m8, cov8
from spectral import *
import matplotlib.pyplot as plt
from PCA import pca


class ArtificialHyperspectralCube:
    """ this class initialize an artificial hyperspectral cube according to the original data
    that was given by the header file.

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

    def __init__(self, header, is_load=False):
        """ this function initialize the class"""
        if is_load is False:
            self.data = open_image(header)
            self.cube = self.data.load(dtype='double').copy()
            self.rows, self.cols, self.bands = self.cube.shape

            self.x_mean = m8(self.cube)
            self.x_cov = cov8(self.cube, self.x_mean)

            self.y, self.x_eigvec = pca(self.cube, self.x_mean, self.x_cov)

            self.y_mean = m8(self.y)
            self.y_cov = cov8(self.y, self.y_mean)

            # self.nu_x = find_nu(self.x_np, self.m8x, self.cov_x, False)
            # self.nu_y = find_nu(self.y_np, self.m8y, self.cov_y, False)

            # Z cube ############
            self.__create_z_cube([2 for i in range(self.bands)])
            self.m8 = m8(self.artificial_data)
            self.cov = cov8(self.artificial_data, self.m8)
            # self.nu = find_nu(self.data, self.m8, self.cov, False)

            # T cube ############
            self.t, _ = pca(self.artificial_data, self.m8, self.cov)
            self.t_mean = m8(self.t)
            self.t_cov = cov8(self.t, self.t_mean)

            # Q cube ############
            self.q = np.zeros(shape=(self.rows, self.cols, self.bands), dtype='double')
            for r in range(self.rows):
                for c in range(self.cols):
                    self.q[r, c, :] = np.matmul(self.x_eigvec, self.t[r, c, :])
            self.q_mean = m8(self.q)
            self.q_cov = cov8(self.q, self.q_mean)
        else:
            self.load_params()

    def __create_z_cube(self, nu_vec):
        self.artificial_data = np.zeros(shape=(self.rows, self.cols, self.bands))
        for band in range(self.bands):
            self.artificial_data[:, :, band] += np.random.standard_t(nu_vec[band], size=(self.rows, self.cols))

        self.m8 = m8(self.artificial_data)
        self.cov = cov8(self.artificial_data, self.m8)

        for band in range(self.bands):
            self.artificial_data[:, :, band] = self.artificial_data[:, :, band] / np.sqrt(self.cov[band, band])
            self.artificial_data[:, :, band] = self.artificial_data[:, :, band] * np.sqrt(self.y_cov[band, band])

        self.artificial_data = np.array(self.artificial_data) + self.y_mean

    def save_params(self):
        np.save('z.npy', self.artificial_data)
        np.save('t.npy', self.t)
        np.save('q.npy', self.q)
        np.save('y.npy', self.y)
        np.save('x.npy', self.cube)
        np.save('x_mean.npy', self.x_mean)
        np.save('x_cov.npy', self.x_cov)
        np.save('y_mean.npy', self.y_mean)
        np.save('y_cov.npy', self.y_cov)
        np.save('t_mean.npy', self.t_mean)
        np.save('t_cov.npy', self.t_cov)
        np.save('q_mean.npy', self.q_mean)
        np.save('q_cov.npy', self.q_cov)
        np.save('x_eigvec.npy', self.x_eigvec)
        np.save('m8.npy', self.m8)
        np.save('cov.npy', self.cov)
        # np.save('nu.npy', self.nu)
        # np.save('nu_x.npy', self.nu_x)
        # np.save('nu_y.npy', self.nu_y)

    def load_params(self):
        self.artificial_data = np.load('z.npy')
        self.t = np.load('t.npy')
        self.q = np.load('q.npy')
        self.y = np.load('y.npy')
        self.cube = np.load('x.npy')
        self.x_mean = np.load('x_mean.npy')
        self.x_cov = np.load('x_cov.npy')
        self.y_mean = np.load('y_mean.npy')
        self.y_cov = np.load('y_cov.npy')
        self.t_mean = np.load('t_mean.npy')
        self.t_cov = np.load('t_cov.npy')
        self.q_mean = np.load('q_mean.npy')
        self.q_cov = np.load('q_cov.npy')
        self.x_eigvec = np.load('x_eigvec.npy')
        self.m8 = np.load('m8.npy')
        self.cov = np.load('cov.npy')
        # self.nu = np.load('nu.npy')
        # self.nu_x = np.load('nu_x.npy')
        # self.nu_y = np.load('nu_y.npy')


if __name__ == "__main__":
    z = ArtificialHyperspectralCube('D1_F12_H2_Cropped_des_Aligned.hdr')
    plt.plot([i for i in range(len(z.nu))], z.nu)
    plt.plot([i for i in range(len(z.m8))], z.m8)
    plt.show()

    pass
