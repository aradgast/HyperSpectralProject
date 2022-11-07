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
    fig_cnt = 0
    bands_vec = [50, 100, 150, 200, 250, 300, 350]
    for band in bands_vec:

        matrix_x = img_np[:, :, band].reshape(rowNum, colNum)
        matrix_x -= m8(matrix_x)
        matrix_x *= 1 / np.sqrt(np.cov(matrix_x.reshape(rowNum*colNum)))

        nu_tmp = 2  # starts from nu value = 2
        simulation = 300  # number of nu values testing
        comp_mat = np.zeros(shape=(rowNum, colNum, simulation))  # save the matrix represent the distribution
        res_vec_p_value = []
        res_vec_stats = []
        x_nu = []
        p_vec = calc_cumulative_dist(matrix_x)
        for i in range(simulation):
            comp_mat[:, :, i] = np.random.standard_t(nu_tmp, size=(rowNum, colNum))
            p_tmp = calc_cumulative_dist(comp_mat[:, :, i])
            test = stats.ks_2samp(p_vec, p_tmp)  # KS test for comparing 2 unknown distribution samples
            if test[1] > 0.05:  # if the hypothesis is correct so add to the result vector
                x_nu.append(nu_tmp)
                res_vec_stats.append(test[0])
                res_vec_p_value.append(test[1])
            nu_tmp += 8 / simulation

        f, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
        ax1.stem(x_nu, res_vec_stats)
        ax1.set_title(f'KS statistics result for band {band}')
        ax1.set_ylabel('max of the distance')
        ax1.grid()

        fig_cnt += 1
        ax2.stem(x_nu, res_vec_p_value)
        ax2.set_title(f'KS p-value result for band {band}')
        ax2.set_xlabel('nu values')
        ax2.set_ylabel('max of the probability')
        ax2.grid()
        fig_cnt += 1
    plt.show()
