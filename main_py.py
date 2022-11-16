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
    # bands_vec = [50, 100, 150, 200, 250, 300, 350]
    bands_vec = [50]
    for band in bands_vec:

        matrix_x = img_np[:, :, band].reshape(rowNum, colNum)
        matrix_x -= m8(matrix_x)
        matrix_x *= 1 / np.sqrt(np.cov(matrix_x.reshape(rowNum*colNum)))
        plt.figure(1)
        plt.hist(matrix_x,bins=100)
        plt.title(f'band {band} histogram')
        nu_tmp = 1  # starts from nu value = 2
        simulation = 100  # number of nu values testing
        comp_mat = np.zeros(shape=(rowNum*colNum//4, simulation))  # save the matrix represent the distribution
        res_vec_p_value = []
        res_vec_stats = []
        x_nu = []

        for i in range(simulation):
            # if i< simulation//2:
            comp_mat[:, i] = np.random.standard_t(nu_tmp, size=rowNum*colNum//4)
            test = stats.ks_2samp(matrix_x.reshape(rowNum * colNum),comp_mat[:, i], alternative='two-sided')  # KS test for comparing 2 unknown distribution samples
            x_nu.append(nu_tmp)
            res_vec_stats.append(test[0])
            res_vec_p_value.append(test[1])
            nu_tmp += 8 / simulation
        f, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x_nu, res_vec_stats, label='t dist')
        axs[0, 0].set_title(f'KS statistics result for band {band}')
        axs[0, 0].set_ylabel('statistics value')
        axs[0, 0].grid()

        fig_cnt += 1
        axs[1, 0].plot(x_nu, res_vec_p_value)
        axs[1, 0].set_title(f'KS p-value result for band {band}')
        axs[1, 0].set_xlabel('nu values')
        axs[1, 0].set_ylabel('p-value')
        axs[1, 0].grid()
S
        axs[0, 1].hist(comp_mat[:,0], bins=500)
        axs[0, 1].set_title(f'histogram for t_dist, nu={x_nu[0]}')

        axs[1, 1].hist(comp_mat[:, -1], bins=100)
        axs[1, 1].set_title(f'histogram for t_dist, nu={x_nu[-1]}')


        fig_cnt += 1
    plt.show()

    n1 = np.random.normal(0,1, 1000)
    n4 = np.random.normal(0,1, 1000)
    n2 = np.random.normal(0,2, 1000)
    n3 = np. random.normal(0,1,500)
    res1 = stats.ks_2samp(n1,n2)
    res2 = stats.ks_2samp(n1,n3)
    res3 = stats.ks_2samp(n1,n4)

    print(f'results for two normal dist with diffrent std -> {res1}')
    print(f'results for two normal dist with same std but diffrent length -> {res2}')
    print(f'results for the same normal dist diffrent sample -> {res3}')