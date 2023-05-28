########################################################################################################################
# All right reserved by BGU, 2023
# Author: Arad Gast, Ido Levokovich
# Date: 03/2023
# Description: this file contains the ArtificialHyperspectralCube class

########################################################################################################################

import numpy as np

import local_mean_covariance
from spectral import *
import matplotlib.pyplot as plt
from PCA import pca
from find_nu import find_nu
from scipy.stats import t as t_dist

PRECISION = np.double

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

    def __init__(self, header, is_load=False, name=None, nu_method='Constant'):
        """ this function initialize the class"""
        self.name = name
        self.nu_method = nu_method
        if is_load is False:
            self.data = open_image(header)
            self.cube = self.data.load(dtype=PRECISION).copy()
            self.rows, self.cols, self.bands = self.cube.shape

            self.x_mean = local_mean_covariance.m8(self.cube)
            self.x_cov = local_mean_covariance.cov8(self.cube, self.x_mean)

            self.y, self.x_eigvec = pca(self.cube, self.x_mean, self.x_cov)

            self.y_mean = local_mean_covariance.m8(self.y)
            self.y_cov = local_mean_covariance.cov8(self.y, self.y_mean)

            # self.nu_x = find_nu(self.cube, self.x_mean, self.x_cov, method=nu_method)
            self.nu_y = find_nu(self.y, self.y_mean, self.y_cov, method=nu_method)

            # Z cube ############
            self.__create_z_cube(self.nu_y)
            # self.m8 = m8(self.artificial_data)
            self.m8 = np.mean(self.artificial_data, (0, 1))
            self.cov = local_mean_covariance.cov8(self.artificial_data, self.m8)
            # self.nu = find_nu(self.data, self.m8, self.cov, method=nu_method)

            # T cube ############
            self.t, _ = pca(self.artificial_data, self.m8, self.cov)
            # self.t_mean = m8(self.t)
            self.t_mean = np.mean(self.t, (0, 1))
            self.t_cov = local_mean_covariance.cov8(self.t, self.t_mean)
            # self.t_nu = find_nu(self.t, self.t_mean, self.t_cov, method=nu_method)

            # Q cube ############
            self.q = np.zeros(shape=(self.rows, self.cols, self.bands), dtype=PRECISION)
            for r in range(self.rows):
                for c in range(self.cols):
                    self.q[r, c, :] = np.matmul(self.x_eigvec, self.t[r, c, :])
            self.q_mean = local_mean_covariance.m8(self.q)
            self.q_mean = np.mean(self.q, (0,1))
            self.q_cov = local_mean_covariance.cov8(self.q, self.q_mean)
            # self.q_nu = find_nu(self.q, self.q_mean, self.q_cov, method=nu_method)

            # G cube ############
            self.g = np.zeros(shape=(self.rows, self.cols, self.bands), dtype=PRECISION)
            for s in range(self.bands):
                self.g[:, :, s] = np.random.normal(loc=0, scale=1, size=(self.rows, self.cols))
                self.g[:, :, s] = self.g[:, :, s] / np.sqrt(np.std(self.g[:, :, s]))
                self.g[:, :, s] = self.g[:, :, s] * np.sqrt(self.y_cov[s, s], dtype=PRECISION)
            self.g += self.y_mean
            # self.g_mean = local_mean_covariance.m8(self.g)
            self.g_mean = np.mean(self.g, (0,1))
            self.g_cov = local_mean_covariance.cov8(self.g, self.g_mean)

            ############################

            self.save_params()
        else:
            self.load_params()

    def __create_z_cube(self, nu_vec):
        self.artificial_data = np.zeros(shape=(self.rows, self.cols, self.bands), dtype=PRECISION)
        self.stats_vec = []
        for band in range(self.bands):
            if nu_vec[band] == 0:
                self.artificial_data[:, :, band] = np.random.normal(loc=0, scale=1, size=(self.rows, self.cols))
            else:
                self.artificial_data[:, :, band] += t_dist.rvs(nu_vec[band], loc=0, scale=1, size=(self.rows, self.cols))
        self.m8 = np.mean(self.artificial_data, (0,1))
        self.cov = local_mean_covariance.cov8(self.artificial_data, self.m8)

        for band in range(self.bands):
            self.artificial_data[:, :, band] = self.artificial_data[:, :, band] / np.sqrt(self.cov[band, band],
                                                                                          dtype=PRECISION)
            self.artificial_data[:, :, band] = self.artificial_data[:, :, band] * np.sqrt(self.y_cov[band, band],
                                                                                          dtype=PRECISION)

        self.artificial_data = np.array(self.artificial_data) + self.y_mean

    def save_params(self):
        """ this function save the parameters of the class in a file"""
        if self.name is not None:
            add = self.name
        else:
            add = ''
        np.save('numpy_saved/x' + add + '.npy', self.cube)
        np.save('numpy_saved/x_mean' + add + '.npy', self.x_mean)
        np.save('numpy_saved/x_cov' + add + '.npy', self.x_cov)
        np.save('numpy_saved/x_eigvec' + add + '.npy', self.x_eigvec)
        # np.save('numpy_saved/x_nu' + add + '.npy', self.nu_x)

        np.save('numpy_saved/y' + add + '.npy', self.y)
        np.save('numpy_saved/y_mean' + add + '.npy', self.y_mean)
        np.save('numpy_saved/y_cov' + add + '.npy', self.y_cov)
        np.save('y_nu' + add + '.npy', self.nu_y)

        np.save('numpy_saved/t' + add + '.npy', self.t)
        np.save('numpy_saved/t_mean' + add + '.npy', self.t_mean)
        np.save('numpy_saved/t_cov' + add + '.npy', self.t_cov)
        # np.save('numpy_saved/t_nu' + add + '.npy', self.t_nu)

        np.save('numpy_saved/q' + add + '.npy', self.q)
        np.save('numpy_saved/q_mean' + add + '.npy', self.q_mean)
        np.save('numpy_saved/q_cov' + add + '.npy', self.q_cov)
        # np.save('q_nu' + add + '.npy', self.q_nu)

        np.save('numpy_saved/z' + add + '.npy', self.artificial_data)
        np.save('numpy_saved/z_mean' + add + '.npy', self.m8)
        np.save('numpy_saved/z_cov' + add + '.npy', self.cov)
        # np.save('numpy_saved/z_nu' + add + '.npy', self.nu)

        # np.save('numpy_saved/stats_vec' + add + '.npy', self.stats_vec)

    def load_params(self):
        """ This function load the parameters from the file"""

        if self.name is not None:
            add = self.name
        else:
            add = ''

        self.cube = np.load('numpy_saved/x' + add + '.npy')
        self.x_mean = np.load('numpy_saved/x_mean' + add + '.npy')
        self.x_cov = np.load('numpy_saved/x_cov' + add + '.npy')
        self.x_eigvec = np.load('numpy_saved/x_eigvec' + add + '.npy')
        # self.nu_x = np.load('numpy_saved/x_nu' + add + '.npy')

        self.y = np.load('numpy_saved/y' + add + '.npy')
        self.y_mean = np.load('numpy_saved/y_mean' + add + '.npy')
        self.y_cov = np.load('numpy_saved/y_cov' + add + '.npy')
        self.nu_y = np.load('numpy_saved/y_nu' + add + '.npy')

        self.t = np.load('numpy_saved/t' + add + '.npy')
        self.t_mean = np.load('numpy_saved/t_mean' + add + '.npy')
        self.t_cov = np.load('numpy_saved/t_cov' + add + '.npy')
        # self.nu_t = np.load('numpy_saved/t_nu' + add + '.npy')

        self.q = np.load('numpy_saved/q' + add + '.npy')
        self.q_mean = np.load('numpy_saved/q_mean' + add + '.npy')
        self.q_cov = np.load('numpy_saved/q_cov' + add + '.npy')
        # self.nu_q = np.load('numpy_saved/q_nu' + add + '.npy')

        self.artificial_data = np.load('numpy_saved/z' + add + '.npy')
        self.m8 = np.load('numpy_saved/z_mean' + add + '.npy')
        self.cov = np.load('numpy_saved/z_cov' + add + '.npy')
        # self.nu = np.load('numpy_saved/z_nu' + add + '.npy')

        # self.stats_vec = np.load('numpy_saved/stats_vec' + add + '.npy')


if __name__ == "__main__":
    z = ArtificialHyperspectralCube('D1_F12_H2_Cropped_des_Aligned.hdr')
    plt.plot([i for i in range(len(z.nu))], z.nu)
    plt.plot([i for i in range(len(z.m8))], z.m8)
    plt.show()

    pass
