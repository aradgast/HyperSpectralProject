########################################################################################################################
# All right reserved by BGU, 2023
# Author: Arad Gast, Ido Levokovich
# Date: 03/2023
# Description: this file contains the ArtificialHyperspectralCube class

########################################################################################################################

import numpy as np
from legends import *
from local_mean_covariance import get_m8, get_cov8
from spectral import *
import matplotlib.pyplot as plt
from PCA import get_pca
from find_nu import find_nu
from scipy.stats import t as t_dist


class HyperSpectralCube:
    """ This class is an object the hold hyperspectral data."""

    def __init__(self, header=None, cube=None):

        if header is not None:
            self.data = open_image(header)
            self.cube = self.data.load(dtype=PRECISION).copy()
        elif cube is not None:
            self.data = None
            self.cube = cube
        else:
            raise Exception('You must provide either a header or a cube')

        self.rows, self.cols, self.bands = self.cube.shape
        self.mean = None
        self.cov = None
        self.nu = None
        self.eigenvectors = None
        self.eigenvalues = None

    def calc_mean(self, method='local'):
        """ This function calculates the mean of the data.
        :param method: the method to calculate the mean
        :return: None
        """
        self.mean = get_m8(self.cube, method=method)

    def calc_cov(self, method='Local'):
        """ This function calculates the covariance of the data.
        :param method: the method to calculate the covariance
        :return: None
        """
        self.cov = get_cov8(self.cube, self.mean, method=method)

    def calc_nu(self, method='Constant2'):
        """ This function calculates the degree of freedom for the data.
        :param method: the method to calculate the degree of freedom
        :return: None
        """
        self.nu = find_nu(self.cube, self.mean, self.cov, method=method)

    def pca_transform(self):
        """ This function transforms the data to the PCA space.
        :return: None
        """
        transformed_cube, self.eigenvectors, self.eigenvalues = get_pca(self.cube, self.mean, self.cov)
        return HyperSpectralCube(cube=transformed_cube)

    def plot(self, band, title=None):
        """ This function plots a specific band of the data.
        :param band: the band to plot
        :param title: the title of the plot
        :return: None
        """
        plt.figure()
        plt.imshow(self.cube[:, :, band], cmap='gray')
        plt.colorbar()
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_mean(self, title=None):
        """ This function plots the mean of the data.
        :param title: the title of the plot
        :return: None
        """
        plt.figure()
        plt.imshow(self.mean, cmap='gray')
        plt.colorbar()
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_cov(self, title=None):
        """ This function plots the covariance of the data.
        :param title: the title of the plot
        :return: None
        """
        plt.figure()
        plt.imshow(self.cov, cmap='gray')
        plt.colorbar()
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_nu(self, title=None):
        """ This function plots the degree of freedom of the data.
        :param title: the title of the plot
        :return: None
        """
        plt.figure()
        plt.semilogy(self.nu)
        plt.colorbar()
        if title is not None:
            plt.title(title)
        plt.xlabel("Band")
        plt.ylabel("DOF")
        plt.show()

    def plot_all(self, title=None):
        """ This function plots the mean, covariance and degree of freedom of the data.
        :param title: the title of the plot
        :return: None
        """
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(self.mean, cmap='gray')
        plt.colorbar()
        plt.title('Mean')
        plt.subplot(1, 3, 2)
        plt.imshow(self.cov, cmap='gray')
        plt.colorbar()
        plt.title('Covariance')
        plt.subplot(1, 3, 3)
        plt.imshow(self.nu, cmap='gray')
        plt.colorbar()
        plt.title('Degree of freedom')
        plt.suptitle(title)
        plt.show()

    def plot_all_bands(self, title=None):
        """ This function plots all the bands of the data.
        :param title: the title of the plot
        :return: None
        """
        plt.figure()
        for i in range(self.bands):
            plt.subplot(1, self.bands, i + 1)
            plt.imshow(self.cube[:, :, i], cmap='gray')
            plt.colorbar()
            plt.title('Band {}'.format(i + 1))
        plt.suptitle(title)
        plt.show()

    def __str__(self):
        """ This function prints the data.
        :return: None
        """
        print(f"This is Hyperspectral cube with {self.bands} bands, {self.rows} rows and {self.cols} columns")
        print(f"The data type is {self.cube.dtype}")
        # print(f"The mean is: \n{self.mean}")
        # print(f"The covariance is: \n{self.cov}")
        if self.nu is not None:
            print(f"The degree of freedom is: \n{self.nu}")
        return ""


class ArtificialHSC(HyperSpectralCube):
    """ this class initialize an artificial hyperspectral cube according to the original data"""

    def __init__(self, original_data, eigenvectors, eigenvalues):
        cube = np.zeros((original_data.rows, original_data.cols, original_data.bands))
        for band in range(original_data.bands):
            if original_data.nu[band] < 2 or original_data.nu[band] > 50:
                cube[:, :, band] = np.random.normal(loc=0, scale=1, size=(original_data.rows, original_data.cols))
            else:
                cube[:, :, band] = t_dist.rvs(original_data.nu[band], loc=0, scale=1, size=(original_data.rows, original_data.cols))
                # cube[:, :, band] *= np.sqrt(cov[band, band])
                # cube[:, :, band] += mean[:, :, band]
        super().__init__(header=None, cube=cube)
        self.calc_mean("global")
        self.calc_cov("global")
        for band in range(self.bands):
            self.cube[:, :, band] *= (np.sqrt(original_data.cov[band, band]) / np.sqrt(self.cov[band, band]))
        self.cube += original_data.mean
        self.calc_mean("global")
        self.calc_cov("global")
        self.cube, self.eigenvectors, self.eigenvalues = get_pca(self.cube, self.mean, self.cov)
        self.calc_mean("global")
        self.calc_cov("global")
        #
        #
        for r in range(self.rows):
            for c in range(self.cols):
                self.cube[r, c, :] = np.matmul(eigenvectors, self.cube[r, c, :]*np.sqrt(eigenvalues))
        #
        self.calc_mean("global")
        self.calc_cov("global")
        # # self.calc_nu("")



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

    def __init__(self, header, statistical_method='local'):
        """ this function initialize the class"""
        self.statistical_method = statistical_method
        self.data = open_image(header)
        self.cube = self.data.load(dtype=PRECISION).copy()
        self.rows, self.cols, self.bands = self.cube.shape

        self.x_mean = get_m8(self.cube)
        self.x_cov = get_cov8(self.cube, self.x_mean)

        self.y, self.x_upscaled_eigvec = get_pca(self.cube, self.x_mean, self.x_cov)

        self.y_mean = get_m8(self.y)
        self.y_cov = get_cov8(self.y, self.y_mean)

        # G cube ############
        self.g = np.zeros(shape=(self.rows, self.cols, self.bands), dtype=PRECISION)
        for s in range(self.bands):
            self.g[:, :, s] = np.random.normal(loc=0, scale=1, size=(self.rows, self.cols))

        self.g_mean = get_m8(self.g, self.statistical_method)
        self.g_cov = get_cov8(self.g, self.g_mean, self.statistical_method)

        for s in range(self.bands):
            self.g[:, :, s] = self.g[:, :, s] / np.sqrt(self.g_cov[s, s])
            self.g[:, :, s] = self.g[:, :, s] * np.sqrt(self.y_cov[s, s])

        self.g += self.y_mean
        self.g_mean = get_m8(self.g, self.statistical_method)
        self.g_cov = get_cov8(self.g, self.g_mean, self.statistical_method)

        # initiate the fields
        self.nu_x = None
        self.nu_y = None
        self.artificial_data = None
        self.cov = None
        self.m8 = None
        self.nu = None
        self.t = None
        self.t_mean = None
        self.t_cov = None
        self.t_nu = None
        self.q = None
        self.q_mean = None
        self.q_cov = None
        self.q_nu = None

    def create_z_cube(self, nu_method='Constant'):
        """ this function creates the artificial hyperspectral cube according to the original data
        that was given by the header file.
        :param nu_method: the method to estimate the df(degree of freedom) for the artificial hyperspectral cube
        :return: None
        """

        # self.nu_x = find_nu(self.cube, self.x_mean, self.x_cov, method=nu_method)
        self.nu_y = find_nu(self.y, self.y_mean, self.y_cov, method=nu_method)

        self.artificial_data = np.zeros(shape=(self.rows, self.cols, self.bands), dtype=PRECISION)
        for band in range(self.bands):
            if self.nu_y[band] == 0:
                self.artificial_data[:, :, band] = np.random.normal(loc=0, scale=1, size=(self.rows, self.cols))
            else:
                self.artificial_data[:, :, band] += t_dist.rvs(self.nu_y[band], loc=0, scale=1,
                                                               size=(self.rows, self.cols))
        self.m8 = get_m8(self.artificial_data, self.statistical_method)
        self.cov = get_cov8(self.artificial_data, self.m8, self.statistical_method)

        for band in range(self.bands):
            self.artificial_data[:, :, band] = self.artificial_data[:, :, band] / np.sqrt(self.cov[band, band],
                                                                                          dtype=PRECISION)
            self.artificial_data[:, :, band] = self.artificial_data[:, :, band] * np.sqrt(self.y_cov[band, band],
                                                                                          dtype=PRECISION)

        self.artificial_data = np.array(self.artificial_data) + self.y_mean

        self.m8 = get_m8(self.artificial_data, self.statistical_method)
        self.cov = get_cov8(self.artificial_data, self.m8, self.statistical_method)
        # self.nu = find_nu(self.data, self.m8, self.cov, method=nu_method)

        # T cube ############
        self.t, _ = get_pca(self.artificial_data, self.m8, self.cov)
        self.t_mean = get_m8(self.t, self.statistical_method)
        self.t_cov = get_cov8(self.t, self.t_mean, self.statistical_method)
        # self.t_nu = find_nu(self.t, self.t_mean, self.t_cov, method=nu_method)

        # Q cube ############
        self.q = np.zeros(shape=(self.rows, self.cols, self.bands), dtype=PRECISION)
        for r in range(self.rows):
            for c in range(self.cols):
                self.q[r, c, :] = np.matmul(self.x_upscaled_eigvec, self.t[r, c, :])
        self.q_mean = get_m8(self.q, self.statistical_method)
        self.q_cov = get_cov8(self.q, self.q_mean, self.statistical_method)
        # self.q_nu = find_nu(self.q, self.q_mean, self.q_cov, method=nu_method)


if __name__ == "__main__":

    pass
