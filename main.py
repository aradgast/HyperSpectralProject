from detection_algo import matched_filter, ace, rx
from ArtificialHyperspectral_class import ArtificialHyperspectralCube, HyperSpectralCube, ArtificialHSC
from plot_detection_algo import plot_stats, calc_stats
import spectral as spy
from local_mean_covariance import get_m8, get_cov8
import numpy as np
import warnings
from scipy.stats import t as t_dist
from find_nu import find_nu
from local_mean_covariance import get_m8, get_cov8
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import datetime

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # NEW FORMATTING
    original_data = HyperSpectralCube('data\self_test_rad.hdr')
    original_data.calc_mean()
    original_data.calc_cov()

    pca_data = original_data.pca_transform()
    pca_data.calc_mean()
    pca_data.calc_cov()
    pca_data.calc_nu("Tyler")

    # artificial data
    artifical_data = ArtificialHSC(pca_data, original_data.eigenvectors, original_data.eigenvalues)

    # MF results
    mf_res_x = matched_filter(0.065, original_data.cube, original_data.mean, original_data.cov,
                              original_data.cube[4, 2].reshape(1, 1, -1))
    mf_res_y = matched_filter(0.065, pca_data.cube, pca_data.mean, pca_data.cov,
                              pca_data.cube[4, 2].reshape(1, 1, -1))
    mf_res_q = matched_filter(0.065, artifical_data.cube, artifical_data.mean, artifical_data.cov,
                              pca_data.cube[4, 2].reshape(1, 1, -1))
    stats_x = calc_stats(mf_res_x[0], mf_res_y[1])
    stats_y = calc_stats(mf_res_y[0], mf_res_x[1])
    stats_q = calc_stats(mf_res_q[0], mf_res_x[1])
    plot_stats([stats_x[0], stats_y[0], stats_q[0]],
               [stats_x[1], stats_y[1], stats_q[1]],
               [stats_x[2], stats_y[2], stats_q[2]],
               [stats_x[3], stats_y[3], stats_q[3]],
               [stats_x[4], stats_y[4], stats_q[4]],
               algo_name='MF', name_of_the_plot='test', save_fig=False)
    print('MF done')



    # header = r'data\self_test_rad.hdr'                       # 'D1_F12_H1_Cropped.hdr', 'blind_test_refl.hdr', 'self_test_rad.hdr', 'bulb_0822-0903.hdr'
    # statistical_method = 'global'                               # 'global', 'local'
    # name = f'RIT_with_{statistical_method}_last_try'      # 'ViaReggio', 'RIT'
    # method = 'MLE'                                      # 'NN', 'MLE', 'Constant2', 'Constant3', 'KS', 'Tyler'
    # # methods = ['NN', 'MLE', 'Constant2', 'KS', 'Tyler']
    # z = ArtificialHyperspectralCube(header, statistical_method=statistical_method)
    # z.create_z_cube(method)
    # mf_res_x = matched_filter(0.065, z.cube, z.x_mean, z.x_cov, z.cube[4, 2].reshape(1, 1, -1))
    # mf_res_q = matched_filter(0.065, z.q, z.q_mean, z.q_cov, z.y[4, 2].reshape(1, 1, -1))
    # stats_x = calc_stats(mf_res_x[0], mf_res_x[1])
    # stats_q = calc_stats(mf_res_q[0], mf_res_q[1])
    # plot_stats(2, [stats_x[0], stats_q[0]],
    #            [stats_x[1], stats_q[1]],
    #            [stats_x[2], stats_q[2]],
    #            [stats_x[3], stats_q[3]],
    #            [stats_x[4], stats_q[4]],
    #            ["Original data", "Artificial data", "Gaussian method"], "MF", name)
    # print("DONE MF")
    #
    # for method in methods:
    #     print("############################################################################################################")
    #     print(f"Method: {method}")
    #     print("############################################################################################################")
    #     z.create_z_cube(method)
    #     print("Matched filter:")
    #     print(f"X peak distance: {np.round(mf_res_x[2], 3)}")
    #     mf_res_q = matched_filter(0.065, z.q, z.q_mean, z.q_cov, z.y[4, 2].reshape(1, 1, -1))
    #     print(f"Q peak distance: {np.round(mf_res_q[2], 3)}")
    #     print(f"G peak distance: {np.round(mf_res_g[2], 3)}")
    #
    #     stats_q = calc_stats(mf_res_q[0], mf_res_q[1])

        # plot_stats(3, [stats_x[0], stats_q[0], stats_g[0]],
        #            [stats_x[1], stats_q[1], stats_g[1]],
        #            [stats_x[2], stats_q[2], stats_g[2]],
        #            [stats_x[3], stats_q[3], stats_g[3]],
        #            [stats_x[4], stats_q[4], stats_g[4]],
        #            ['Original data', 'Artificial data', 'Gaussian method'], 'MF', name, method)
    #
    ##################################################################################################################
    # simulation for checking the DOF estimation methods.

    # size_of_simulation = 150
    # size_of_matrix = 300
    # methods = ['NN', 'MLE', 'Tyler', 'KS']
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
    # plt.semilogy([_ for _ in range(size_of_simulation)], true_nu, label='True nu', color='black', linestyle='--', linewidth=2)
    # for method in methods:
    #     print(f"Method: {method}")
    #     nu = find_nu(cube, m8_cube, cov8_cube, method)
    #     plt.semilogy([_ for _ in range(size_of_simulation)], nu, label=method)
    #     print(f"Done with: {method}")
    # plt.title("DOF estimation with different methods")
    # plt.legend()
    # plt.grid()
    # plt.savefig("plots/DOF estimation with different methods.png")
    # plt.show()

    ##################################################################################################################
    # simulation for checking the DOF estimation methods. now increasing the real DOF by 1 in each simulation
    # monte_carlo = 30
    # lenght_of_simulation = 100
    # true_label = np.linspace(2, 30, lenght_of_simulation)
    # size_of_matrix = 200
    # methods = ['NN', 'MLE', 'Tyler', 'KS']
    # estimated_label = {_: np.zeros((lenght_of_simulation, 1)) for _ in methods}
    # cube = np.zeros((size_of_matrix, size_of_matrix, lenght_of_simulation)).astype(np.single)
    # for _ in range(monte_carlo):
    #     for band, nu in enumerate(true_label):
    #         cube[:, :, band] = t_dist.rvs(nu, loc=0, scale=1, size=(size_of_matrix, size_of_matrix)).astype(np.single)
    #     m8 = get_m8(cube)
    #     cov8 = get_cov8(cube, m8)
    #     print("Done with creating the data.")
    #     for method in methods:
    #         print(f"Method: {method}")
    #         estimated_label[method] += (np.array(find_nu(cube, m8, cov8, method))/ monte_carlo).reshape(-1, 1)
    #         print(f"Done with: {method}")
    # # plot the results
    # plt.figure()
    # for method in methods:
    #     y = np.array(estimated_label[method]).reshape(-1)
    #     plt.plot(true_label, y, label=method)
    # plt.title("DOF estimation with different methods")
    # plt.xlabel("True DOF")
    # plt.ylabel("Estimated DOF")
    # plt.legend()
    # plt.grid()
    # plt.savefig("plots/DOF estimation with different methods for increasing Nu.png")
    # plt.show()
    # print("Done with the simulation.")

    ##################################################################################################################
    # show the hyperspectral image
    # header = "data/blind_test_refl.hdr"
    # data = spy.open_image(header)
    # cube = data.load()
    # view = spy.imshow(cube, (15, 7, 0), alpha=0.95)
    # plt.show()

    ##################################################################################################################
    # results for final report
    # statistical_method = 'global'                               # 'global', 'local'
    # methods = ['NN', 'MLE', 'Constant2', 'KS', 'Tyler']
    # mf_res_q = {k:None for k in methods}
    # stats_q = {k:None for k in methods}
    #
    # for header in ['D1_F12_H1_Cropped.hdr', 'blind_test_refl.hdr']:
    #     z = ArtificialHyperspectralCube("data/" + header, statistical_method=statistical_method)
    #     mf_res_x = matched_filter(0.065, z.cube, z.x_mean, z.x_cov, z.cube[4, 2].reshape(1, 1, -1))
    #     stats_x = calc_stats(mf_res_x[0], mf_res_x[1])
    #     mf_res_g = matched_filter(0.065, z.g, z.g_mean, z.g_cov, z.y[4, 2].reshape(1, 1, -1))
    #     stats_g = calc_stats(mf_res_g[0], mf_res_g[1])
    #     for method in methods:
    #         z.create_z_cube(method)
    #         mf_res_q[method] = matched_filter(0.065, z.q, z.q_mean, z.q_cov, z.y[4, 2].reshape(1, 1, -1))
    #         stats_q[method] = calc_stats(mf_res_q[method][0], mf_res_q[method][1])
    #     plot the results
        # plot_stats(2+len(methods), [stats_x[0], stats_g[0], stats_q['NN'][0], stats_q['MLE'][0], stats_q['Constant2'][0], stats_q['KS'][0], stats_q['Tyler'][0]],
        # [stats_x[1], stats_g[1], stats_q['NN'][1], stats_q['MLE'][1], stats_q['Constant2'][1], stats_q['KS'][1], stats_q['Tyler'][1]],
        # [stats_x[2], stats_g[2], stats_q['NN'][2], stats_q['MLE'][2], stats_q['Constant2'][2], stats_q['KS'][2], stats_q['Tyler'][2]],
        # [stats_x[3], stats_g[3], stats_q['NN'][3], stats_q['MLE'][3], stats_q['Constant2'][3], stats_q['KS'][3], stats_q['Tyler'][3]],
        # [stats_x[4], stats_g[4], stats_q['NN'][4], stats_q['MLE'][4], stats_q['Constant2'][4], stats_q['KS'][4], stats_q['Tyler'][4]],
       # ["Original data", "Gaussian method", "NN", "MLE", "Constant2", "KS", "Tyler"], "MF", f"plots/Results for final report_{header}.png")

##################################################################################################################
    # Simulation : create artificial cube using MLE method and plot the DOF vector results.
    # headers = ["D1_F12_H1_Cropped.hdr", 'self_test_rad.hdr']                      # 'D1_F12_H1_Cropped.hdr', 'blind_test_refl.hdr', 'self_test_rad.hdr', 'bulb_0822-0903.hdr'
    # statistical_method = 'local'                               # 'global', 'local'
    # method = 'MLE'
    # header = "data/" + headers[1]
    # z = ArtificialHyperspectralCube(header, statistical_method=statistical_method)
    # z.create_z_cube(method)
    # nu_z = find_nu(z.artificial_data, z.m8, z.cov, method)
    # nu_q = find_nu(z.t, z.t_mean, z.t_cov, method)
    # plt.figure()
    # plt.plot(z.nu_y, label="estimated on Y")
    # plt.plot(nu_z, label="estimated on Z")
    # plt.title(f"DOF estimation using {method} method")
    # plt.xlabel("Band number")
    # plt.ylabel("DOF")
    # plt.legend()
    # plt.grid()
    # plt.savefig(f"plots/DOF estimation values linear scale using {method} method_RIT.png")
    # plt.show()

    # plt.figure()
    # plt.semilogy(z.nu_y, label="estimated on Y")
    # plt.semilogy(nu_z, label="estimated on Z")
    # plt.title(f"DOF estimation using {method} method")
    # plt.xlabel("Band number")
    # plt.ylabel("DOF")
    # plt.legend()
    # plt.grid()
    # plt.savefig(f"plots/DOF estimation values log scale using {method} method_RIT.png")
    # plt.show()
    # print("Done with the simulation.")
##################################################################################################################
    # Simulation : create artificial cube using diffreent values of constant DOF and plot the AUC results.
    # headers = ["D1_F12_H1_Cropped.hdr", 'self_test_rad.hdr']                      # 'D1_F12_H1_Cropped.hdr', 'blind_test_refl.hdr', 'self_test_rad.hdr', 'bulb_0822-0903.hdr'
    # statistical_method = 'local'                               # 'global', 'local'
    # methods = ['Constant2', 'Constant2.5', 'Constant3', 'Constant3.5', 'Constant4', 'Constant4.5', 'Constant5', 'Constant5.5', 'Constant6', 'Constant6.5', 'Constant7', 'Constant7.5', 'Constant8']
    # header = "data/" + headers[1]
    # z = ArtificialHyperspectralCube(header, statistical_method=statistical_method)
    # mf_res_x = matched_filter(0.065, z.cube, z.x_mean, z.x_cov, z.cube[4, 2].reshape(1, 1, -1))
    # stats_x = calc_stats(mf_res_x[0], mf_res_x[1])
    # fpr = stats_x[2]
    # tpr = stats_x[3]
    # idx = len(fpr[fpr <= 0.01])
    # roc_auc_x = auc(fpr[:idx], tpr[:idx])
    # relativec_err = []
    # for method in methods:
    #     z.create_z_cube(method)
    #     mf_res_q = matched_filter(0.065, z.q, z.q_mean, z.q_cov, z.y[4, 2].reshape(1, 1, -1))
    #     stats_q = calc_stats(mf_res_q[0], mf_res_q[1])
    #     fpr = stats_q[2]
    #     tpr = stats_q[3]
    #     idx = len(fpr[fpr <= 0.01])
    #     roc_auc_q = auc(fpr[:idx], tpr[:idx])
    #     relativec_err.append((np.abs(roc_auc_q - roc_auc_x) / roc_auc_x) * 100)
    # plt.figure()
    # plt.plot([2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,6, 6.5, 7, 7.5, 8], relativec_err)
    # plt.title(f"Relative error in AUC for different constant DOF values")
    # plt.xlabel("Constant DOF value")
    # plt.ylabel("Relative error in AUC (%)")
    # plt.grid()
    # plt.savefig(f"plots/Relative error in AUC for different constant DOF values_RIT.png")
    # plt.show()
    # print("Done with the simulation.")
# ##################################################################################################################
#     # Simulation : show histogram of real and artificial data using constant DOF method.
#     headers = ["D1_F12_H1_Cropped.hdr", 'self_test_rad.hdr']                      # 'D1_F12_H1_Cropped.hdr', 'blind_test_refl.hdr', 'self_test_rad.hdr', 'bulb_0822-0903.hdr'
#     name = ["ViaReggio", "RIT"]
#     statistical_method = 'local'                               # 'global', 'local'
#     method = "MLE"
#     header = "data/" + headers[1]
#     z = ArtificialHyperspectralCube(header, statistical_method=statistical_method)
#     z.create_z_cube(method)
#     nu_x = find_nu(z.cube, z.x_mean, z.x_cov, method)
#     nu_z = find_nu(z.artificial_data, z.m8, z.cov, method)
#     nu_q = find_nu(z.q, z.q_mean, z.q_cov, method)
#     plt.figure()
#     plt.semilogy(nu_x, label="real data before PCA")
#     plt.semilogy(z.nu_y, label="real data after PCA")
#     plt.semilogy(nu_z, label="artificial data before PCA")
#     plt.semilogy(nu_q, label="artificial data after PCA")
#     plt.title(f"Comparing Nu values for real and artificial data")
#     plt.xlabel("Band number")
#     plt.ylabel("Nu - log scale")
#     plt.legend()
#     plt.grid()
#     plt.savefig(f"plots/Comparing Nu values for real and artificial data_RIT.png")
#     plt.show()
#
#     for band in [0, 9, 19, 49, 99]:
#         plt.figure()
#         plt.subplots(1, 2, figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.hist((z.q[:, :, band]-z.q_mean[:, :, band]).flatten(), bins=100, label=f"band: {band}")
#         plt.title("Histogram of artificial data - final cube")
#         plt.xlabel("Pixel value")
#         plt.ylabel("Number of pixels")
#         plt.legend()
#         plt.grid()
#         plt.subplot(1, 2, 2)
#         plt.hist((z.cube[:, :, band]-z.x_mean[:, :, band]).flatten(), bins=100, label=f"band: {band}")
#         plt.title("Histogram of real data - m8")
#         plt.xlabel("Pixel value")
#         plt.ylabel("Number of pixels")
#         plt.legend()
#         plt.grid()
#         plt.savefig(f"plots/Histogram of real and artificial data using {method} method_{name[1]}_band_{band+1}.png")
#         plt.show()
#     print("Done with the simulation.")
#################################################################################################################
    # Simulation : check if the DOF estimation is correct for diffreent values of constant std.
    # sample_size = 20000
    # smaple_1 = t_dist.rvs(2, loc=0, scale=1, size=sample_size)
    # smaple_10 = t_dist.rvs(2, loc=0, scale=10, size=sample_size)
    # smaple_100 = t_dist.rvs(2, loc=0, scale=100, size=sample_size)
    # stats_1 = t_dist.fit(smaple_1)
    # stats_10 = t_dist.fit(smaple_10)
    # stats_100 = t_dist.fit(smaple_100)
    # print("for std = 1, DOF = ", stats_1[0])
    # print("for std = 10, DOF = ", stats_10[0])
    # print("for std = 100, DOF = ", stats_100[0])
    # print("Done with the simulation.")