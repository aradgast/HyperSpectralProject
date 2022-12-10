import spectral as spy
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from find_nu import find_nu
from m8 import m8
from MF import matched_filter
from ArtificialHyperspectral_class import ArtificialHyperspectralCube

if __name__ == "__main__":
    z = ArtificialHyperspectralCube('D1_F12_H2_Cropped_des_Aligned.hdr')

    z_axis, hist_z_mf_wt, hist_z_mf_nt, z_inv_cumulative_wt,z_inv_cumulative_nt  = matched_filter(0.065, z.data, z.m8, z.cov, (5,3), True, 'Z')
    x_axis, hist_x_mf_wt, hist_x_mf_nt, x_inv_cumulative_wt,x_inv_cumulative_nt = matched_filter(0.065, z.x_np, z.m8x, z.cov_x, (5,3), True, 'X')
    y_axis, hist_y_mf_wt, hist_y_mf_nt, y_inv_cumulative_wt,y_inv_cumulative_nt = matched_filter(0.065, z.y_np, z.m8y, z.cov_y, (5,3), True, 'Y')
    plt.show()

    plt.figure(1)
    plt.plot(hist_x_mf_wt[1][1:], hist_x_mf_wt[0], hist_x_mf_nt[1][1:], hist_x_mf_nt[0], '--')
    plt.plot(hist_y_mf_wt[1][1:], hist_y_mf_wt[0], hist_y_mf_nt[1][1:], hist_y_mf_nt[0], '--')
    plt.plot(hist_z_mf_wt[1][1:], hist_z_mf_wt[0], hist_z_mf_nt[1][1:], hist_z_mf_nt[0], '--')
    plt.title('MF - histogram results for X&Y&Z')
    plt.grid()
    plt.legend(['WT - X', 'NT - X', 'WT - Y', 'NT - Y', 'WT - Z', 'NT - Z'])

    plt.figure(2)
    plt.plot(x_axis, x_inv_cumulative_wt, x_axis, x_inv_cumulative_nt, '--')
    plt.plot(y_axis, y_inv_cumulative_wt, y_axis, y_inv_cumulative_nt, '--')
    plt.plot(z_axis, z_inv_cumulative_wt, z_axis, z_inv_cumulative_nt, '--')
    plt.title('MF - inverse cumulative probability for X&Y&Z')
    plt.grid()
    plt.legend(['WT - X', 'NT - X', 'WT - Y', 'NT - Y', 'WT - Z', 'NT - Z'])

    plt.figure(3)
    plt.plot(x_inv_cumulative_nt, x_inv_cumulative_wt)
    plt.plot(y_inv_cumulative_nt, y_inv_cumulative_wt)
    plt.plot(z_inv_cumulative_nt, z_inv_cumulative_wt)
    plt.xlim([0, 0.01])
    plt.title('MF - ROC curve with limited pfa for X&Y&Z')
    plt.xlabel('Pfa')
    plt.ylabel('Pd')
    plt.grid()
    plt.legend(['X', 'Y', 'Z'])

    pass    #for debugging




    ###############################################################
    # img = spy.open_image('D1_F12_H2_Cropped_des_Aligned.hdr')
    # img_np = img.open_memmap().copy()  # for working with numpy
    #
    # rowNum, colNum, bandsNum = img.shape
    # # bands_vec = [50, 100, 150, 200, 250, 300, 350]
    # bands_vec = [150]
    # for band in bands_vec:
    #
    #     matrix_x = img_np[:, :, band].reshape(rowNum, colNum)
    #     matrix_x -= m8(matrix_x)
    #     matrix_x = matrix_x.reshape(rowNum * colNum)
    #     matrix_x *= 1 / np.std(matrix_x)
    #     nu_tmp = 2  # starts from nu value = 2
    #     simulation = 200  # number of nu values testing
    #     comp_mat = np.zeros(shape=(rowNum*colNum, simulation))  # save the matrix represent the distribution
    #     res_vec_p_value = []
    #     res_vec_stats = []
    #     x_nu = []
    #
    #     for i in range(simulation):
    #         comp_mat[:, i] = np.random.standard_t(nu_tmp, size=rowNum*colNum)
    #         comp_mat[:, i] *= 1/np.sqrt(np.cov(comp_mat[:, i]))
    #         comp_mat[:, i] -= np.mean(comp_mat[:, i])
    #         test = stats.ks_2samp(matrix_x,comp_mat[:, i], alternative='two-sided')  # KS test for comparing 2 unknown distribution samples
    #         x_nu.append(nu_tmp)
    #         res_vec_stats.append(test[0])
    #         res_vec_p_value.append(test[1])
    #         nu_tmp += 3 / simulation
    #     f, axs = plt.subplots(2, 2)
    #     axs[0, 0].plot(x_nu, res_vec_stats, label='t dist')
    #     axs[0, 0].set_title(f'KS statistics result for band {band}')
    #     axs[0, 0].set_ylabel('statistics value')
    #     axs[0, 0].grid()
    #
    #     axs[1, 0].plot(x_nu, res_vec_p_value)
    #     axs[1, 0].set_title(f'KS p-value result for band {band}')
    #     axs[1, 0].set_xlabel('nu values')
    #     axs[1, 0].set_ylabel('p-value')
    #     axs[1, 0].grid()
    #     idx = np.array(res_vec_stats).argmin()
    #     axs[0, 1].hist(comp_mat[:,idx], bins=800)
    #     axs[0, 1].set_title(f'histogram for t_dist for the lowest statistics, nu={"{:.3f}".format(x_nu[idx])}')
    #     # axs[0, 1].set_xlim([-20, 20])
    #
    #     axs[1, 1].hist(matrix_x, bins=800)
    #     axs[1, 1].set_title(f'histogram for band={band}')
    #     # axs[1, 1].set_xlim([-20, 20])


    # f.tight_layout()
    # plt.show()

    # n1 = np.random.normal(0,1, 100)
    # n4 = np.random.normal(0,1, 100)
    # n2 = np.random.normal(0,2, 100)
    # n3 = np. random.normal(0,1,50)
    # res1 = stats.ks_2samp(n1,n2)
    # res2 = stats.ks_2samp(n1,n3)
    # res3 = stats.ks_2samp(n1,n4)
    # res4 = stats.ks_2samp(n1,n1)
    #
    # print(f'results for two normal dist with different std -> {res1}')
    # print(f'results for two normal dist with same std but different length -> {res2}')
    # print(f'results for the same normal dist different sample -> {res3}')
    # print(f'results for the same normal dist same sample -> {res4}')