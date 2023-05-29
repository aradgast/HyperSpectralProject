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
    header = 'self_test_rad.hdr'        # 'D1_F12_H1_Cropped.hdr', 'blind_test_refl.hdr', 'self_test_rad.hdr', 'bulb_0822-0903.hdr'
    name = 'RIT_with_gaussian'                      # 'ViaReggio', 'RIT'
    # method = 'Constant2'                     # 'NN', 'MLE', 'Constant2', 'Constant3', 'KS', 'Tyler'
    methods = ['NN', 'MLE', 'Constant2', 'KS', 'Tyler']
    z = ArtificialHyperspectralCube(header)
    mf_res_x = matched_filter(0.065, z.cube, z.x_mean, z.x_cov, z.cube[4, 2].reshape(1, 1, -1))
    mf_res_g = matched_filter(0.065, z.g, z.g_mean, z.g_cov, z.y[4, 2].reshape(1, 1, -1))
    stats_x = calc_stats(mf_res_x[0], mf_res_x[1])
    stats_g = calc_stats(mf_res_g[0], mf_res_g[1])

    for method in methods:
        print("############################################################################################################")
        print(f"Method: {method}")
        print("############################################################################################################")
        z.create_z_cube(method)
        print("Matched filter:")
        print(f"X peak distance: {np.round(mf_res_x[2], 3)}")
        mf_res_q = matched_filter(0.065, z.q, z.q_mean, z.q_cov, z.y[4, 2].reshape(1, 1, -1))
        print(f"Q peak distance: {np.round(mf_res_q[2], 3)}")
        print(f"G peak distance: {np.round(mf_res_g[2], 3)}")

        stats_q = calc_stats(mf_res_q[0], mf_res_q[1])
        #
        plot_stats(3, [stats_x[0], stats_q[0], stats_g[0]],
                   [stats_x[1], stats_q[1], stats_g[1]],
                   [stats_x[2], stats_q[2], stats_g[2]],
                   [stats_x[3], stats_q[3], stats_g[3]],
                   [stats_x[4], stats_q[4], stats_g[4]],
                   ['Original data', 'Artificial data', 'Gaussian method'], 'MF', name, method)

    ###################################################################################################################
    # simulation for checking the DOF estimation methods.

    # size_of_simulation = 150
    # size_of_matrix = 300
    # methods = ['NN', 'MLE', 'Tyler']
    # true_nu = []
    # cube = np.zeros((size_of_matrix, size_of_matrix, size_of_simulation)).astype(np.single)
    # for s in range(size_of_simulation):
    #     tmp_nu = np.random.uniform(2, 30)
    #     cube[:, :, s] = t_dist.rvs(tmp_nu, loc=0, scale=1, size=(size_of_matrix, size_of_matrix)).astype(np.single)
    #     true_nu.append(tmp_nu)
    # m8_cube = get_m8(cube)
    # cov8_cube = get_cov8(cube, m8_cube)
    # print("Done with creating the data.")
    # plt.figure()
    # plt.plot([_ for _ in range(size_of_simulation)], true_nu, label='True nu')
    # for method in methods:
    #     print(f"Method: {method}")
    #     nu = find_nu(cube, m8_cube, cov8_cube, method)
    #     plt.plot([_ for _ in range(size_of_simulation)], nu, label=method)
    #     print(f"Done with: {method}")
    # plt.title("DOF estimation with different methods")
    # plt.legend()
    # plt.grid()
    # plt.savefig("plots/DOF estimation with different methods.png")
    # plt.show()



