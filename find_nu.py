import numpy as np
import scipy.stats as stats


def find_nu(cube, mean_matrix, cov):
    '''return the nu vector for the given cube.
    to possible methodas for finding nu:
    1. estimate nu based on james teiler formula
    2. create a cube of t-distribution and find the nu for each band
    '''
    # 1. estimate nu based on james teiler formula
    # bands = cube.shape[2]
    # nu = np.zeros((bands, 1))
    # for i in range(bands):
    #     r = np.abs(cube[:, :, i] / np.sqrt(cov[i, i]))
    #     k = np.mean(np.power(r, 3)) / np.mean(r)
    #     nu[i] = 2 + k / (k - 2)

    # 2. create a cube of t-distribution and find the nu for each band

    comp_mat = np.zeros(shape=(cube.rowNum * cube.colNum, cube.bandNum))  # save the matrix represent the distribution
    nu_init = 2  # starts from nu value = 2
    simulation = 200  # number of nu values testing
    nu_vec = np.zeros(shape=(simulation, 1))
    nu_res = np.zeros((cube.bandNum, 1))
    statiscis_result = np.zeros((simulation, 1))

    for i in range(simulation):
        comp_mat[:, i] = np.random.standard_t(nu_init, size=cube.rowNum * cube.colNum)
        comp_mat[:, i] *= 1 / np.std(comp_mat[:, i])
        comp_mat[:, i] -= np.mean(comp_mat[:, i])
        nu_vec[i] = nu_init
        nu_init += 3 / simulation

    for band in range(cube.bandNum):
        for sim in range(simulation):
            test = stats.ks_2samp(cube.data[:, :, band].reshape(cube.rowNum * cube.colNum), comp_mat[:, sim], alternative='two-sided')  # KS test for comparing 2 unknown distribution samples
            statiscis_result[sim] = test[0]
        nu_res[band] = nu_vec[np.argmin(statiscis_result)]

    return nu_res
