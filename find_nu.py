############################################################################################################
# All right reserved by BGU, 2023
# Author: Arad Gast, Ido Levokovich
# Date: 03/2023
# Description: This file contains the find_nu function

#############################################################################################################
import numpy as np
from scipy.stats import ks_2samp


def find_nu(cube, mean_matrix, cov, method):
    '''return the nu vector for the given cube.
    to possible methodas for finding nu:
    1. estimate nu based on james teiler formula
    2. create a cube of t-distribution and find the nu for each band
    '''
    # 1. estimate nu based on james teiler formula
    if not method:
        bands = cube.shape[2]
        nu = np.zeros((bands, 1))
        for i in range(bands):
            r = np.abs(cube[:, :, i] / np.sqrt(cov[i, i]))
            k = np.mean(np.power(r, 3)) / np.mean(r)
            nu[i] = 2 + k / (k - 2)

    # 2. create a cube of t-distribution and find the nu for each band
    else:
        comp_mat = np.zeros(shape=(cube.shape[0] * cube.shape[1], cube.shape[2]))  # save the matrix represent the distribution
        nu_init = 2  # starts from nu value = 2
        simulation = 200  # number of nu values testing
        nu_vec = np.zeros(shape=(simulation, 1))
        nu = np.zeros((cube.shape[2], 1))
        statiscis_result = np.zeros((simulation, 1))

        for i in range(simulation):
            comp_mat[:, i] = np.random.standard_t(nu_init, size=cube.shape[0] * cube.shape[1])
            comp_mat[:, i] *= 1 / np.std(comp_mat[:, i])
            comp_mat[:, i] -= np.mean(comp_mat[:, i])
            nu_vec[i] = nu_init
            nu_init += 3 / simulation

        for band in range(cube.shape[2]):
            for sim in range(simulation):
                test = ks_2samp(cube[:, :, band].reshape(cube.shape[0] * cube.shape[1]), comp_mat[:, sim], alternative='two-sided')  # KS test for comparing 2 unknown distribution samples
                statiscis_result[sim] = test[0]
            nu[band] = nu_vec[np.argmin(statiscis_result)]

    return nu
