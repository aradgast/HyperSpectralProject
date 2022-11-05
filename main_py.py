import spectral as spy
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from find_nu import find_nu
from m8 import m8
from ArtificialHyperspectral_class import ArtificialHyperspectralCube

if __name__ == "__main__":
    # z = ArtificialHyperspectralCube('D1_F12_H2_Cropped_des_Aligned.hdr')
    # plt.figure(1)
    # plt.plot([i for i in range(len(z.nu))], z.nu, title='nu')
    # plt.figure(2)
    # plt.plot([i for i in range(len(z.m8))], z.m8, title='m8')
    # plt.figure(3)
    # plt.imshow(z.data[:, :, 0].reshape(z.rowNum, z.colNum), cmap='gist_rainbow', title="'0' band")
    # plt.show()
    # Loading the data
    def calc_cumulative_dist(mat):
        """this function get as parameter a matrix,
            and return the cumulative distribution vector"""
        my_hist = np.histogram(mat, 100)
        my_p_vec = np.zeros(shape=len(my_hist[0]))
        s = 0
        for i in range(len(my_p_vec)):
            s += my_hist[0][i] / sum(my_hist[0])
            my_p_vec[i] = s
        return my_p_vec


    ###############################################################
    img = spy.open_image('D1_F12_H2_Cropped_des_Aligned.hdr')
    img_np = img.open_memmap().copy()  # for working with numpy

    rowNum, colNum, bandsNum = img.shape

    matrix_x = img_np[:, :, 200].reshape(rowNum, colNum)
    matrix_x -= m8(matrix_x)
    matrix_x *= 1 / np.sqrt(np.cov(matrix_x.reshape(rowNum*colNum)))

    nu_tmp = 2  # starts from nu value = 2
    simulation = 100  # number of nu values testing
    comp_mat = np.zeros(shape=(rowNum, colNum, simulation))  # save the matrix represent the distribution
    res_vec = []
    x_nu = []
    p_vec = calc_cumulative_dist(matrix_x)
    for i in range(simulation):
        comp_mat[:, :, i] = np.random.standard_t(nu_tmp, size=(rowNum, colNum))
        p_tmp = calc_cumulative_dist(comp_mat[:, :, i])
        test = stats.ks_2samp(p_vec, p_tmp)  # KS test for comparing 2 unknown distribution samples
        if test[1] > 0.05:  # if the hypothesis is correct so add to the result vector
            x_nu.append(nu_tmp)
            res_vec.append(test[0])
        nu_tmp += 8 / simulation

    plt.figure()
    plt.stem(x_nu, res_vec)
    plt.title('KS test result')
    plt.xlabel('nu values')
    plt.ylabel('max of the distance')
    plt.grid()
    plt.show()
