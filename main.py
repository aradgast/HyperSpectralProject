from detection_algo import matched_filter, ace, rx
from ArtificialHyperspectral_class import ArtificialHyperspectralCube
from plot_detection_algo import plot_stats, calc_stats
import spectral as spy
from local_mean_covariance import get_m8, get_cov8
import numpy as np
import warnings
from scipy.stats import t as t_dist
from find_nu import find_nu
from local_mean_covariance import get_m8, get_cov8
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # header = 'self_test_rad.hdr'        # 'D1_F12_H1_Cropped.hdr', 'blind_test_refl.hdr', 'self_test_rad.hdr', 'bulb_0822-0903.hdr'
    # name = 'RIT_with_gaussian_andGlobalMean'                      # 'ViaReggio', 'RIT'
    # # method = 'Constant2'                     # 'NN', 'MLE', 'Constant2', 'Constant3', 'KS', 'Tyler'
    # methods = ['NN', 'Constant2', 'Tyler']
    # for method in methods:
    #     print("############################################################################################################")
    #     print(f"Method: {method}")
    #     print("############################################################################################################")
    #     z = ArtificialHyperspectralCube(header, False, name, nu_method=method)
    #
    #
    #     print("Matched filter:")
    #     mf_res_x = matched_filter(0.065, z.cube, z.x_mean, z.x_cov, z.cube[4, 2].reshape(1, 1, -1))
    #     print(f"X peak distance: {np.round(mf_res_x[2], 3)}")
    #
    #     mf_res_y = matched_filter(0.065, z.y, z.y_mean, z.y_cov, z.y[4, 2].reshape(1, 1, -1))
    #     print(f"Y peak distance: {np.round(mf_res_y[2], 3)}")
    #
    #     mf_res_z = matched_filter(0.065, z.artificial_data, z.m8, z.cov, z.y[4, 2].reshape(1, 1, -1))
    #     print(f"Z peak distance: {np.round(mf_res_z[2], 3)}")
    #
    #     mf_res_t = matched_filter(0.065, z.t, z.t_mean, z.t_cov, z.y[4, 2].reshape(1, 1, -1))
    #     print(f"T peak distance: {np.round(mf_res_t[2], 3)}")
    #
    #     mf_res_q = matched_filter(0.065, z.q, z.q_mean, z.q_cov, z.y[4, 2].reshape(1, 1, -1))
    #     print(f"Q peak distance: {np.round(mf_res_q[2], 3)}")
    #
    #     mf_res_g = matched_filter(0.065, z.g, z.g_mean, z.g_cov, z.y[4, 2].reshape(1, 1, -1))
    #     print(f"G peak distance: {np.round(mf_res_g[2], 3)}")
    #
    #     stats_x = calc_stats(mf_res_x[0], mf_res_x[1])
    #     stats_y = calc_stats(mf_res_y[0], mf_res_y[1])
    #     stats_z = calc_stats(mf_res_z[0], mf_res_z[1])
    #     stats_t = calc_stats(mf_res_t[0], mf_res_t[1])
    #     stats_q = calc_stats(mf_res_q[0], mf_res_q[1])
    #     stats_g = calc_stats(mf_res_g[0], mf_res_g[1])
    #     #
    #     plot_stats(6, [stats_x[0], stats_y[0], stats_z[0], stats_t[0], stats_q[0], stats_g[0]],
    #                [stats_x[1], stats_y[1], stats_z[1], stats_t[1], stats_q[1], stats_g[1]],
    #                [stats_x[2], stats_y[2], stats_z[2], stats_t[2], stats_q[2], stats_g[2]],
    #                [stats_x[3], stats_y[3], stats_z[3], stats_t[3], stats_q[3], stats_g[3]],
    #                [stats_x[4], stats_y[4], stats_z[4], stats_t[4], stats_q[4], stats_g[4]],
    #                ['X', 'Y', 'Z', 'T', 'Q', 'G'], 'MF', name, method)


    # simulation for checking the DOF estimation methods.

    # first, creating the data.
    relative_dof = 10
    size_of_simulation = 100
    size_of_matrix = 300
    methods = ['NN', 'MLE', 'Constant2', 'Constant3', 'KS', 'Tyler']
    cube = t_dist.rvs(relative_dof, loc=0, scale=1, size=(size_of_matrix, size_of_matrix, size_of_simulation))
    m8_cube = get_m8(cube)
    cov8_cube = get_cov8(cube, m8_cube)
    plt.figure()
    for method in methods:
        nu = find_nu(cube, m8_cube, cov8_cube, method)
        plt.plot([_ for _ in range(size_of_simulation)], nu, label=method)
    plt.title("DOF estimation in different methods")
    plt.legend()
    plt.grid()
    plt.savefig("plots/DOF estimation in different methods.png")
    plt.show()



