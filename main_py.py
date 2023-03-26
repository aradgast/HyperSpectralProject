from MF import matched_filter
from ArtificialHyperspectral_class import ArtificialHyperspectralCube
from plot_detection_algo import plot_stats, calc_stats
import spectral as spy
from local_mean_covariance import m8, cov8
import numpy as np

if __name__ == "__main__":
    header = 'D1_F12_H1_Cropped.hdr'
    # z = ArtificialHyperspectralCube(header)
    z = ArtificialHyperspectralCube(header, True)

    mf_res_z = matched_filter(0.065, z.artificial_data, z.m8, z.cov, (4, 2))
    print(f"Z peak distance: {np.round(mf_res_z[2], 3)}")

    mf_res_x = matched_filter(0.065, z.cube, z.x_mean, z.x_cov, (4, 2))
    print(f"X peak distance: {np.round(mf_res_x[2], 3)}")

    mf_res_y = matched_filter(0.065, z.y, z.y_mean, z.y_cov, (4, 2))
    print(f"Y peak distance: {np.round(mf_res_y[2], 3)}")

    mf_res_t = matched_filter(0.065, z.t, z.t_mean, z.t_cov, (4, 2))
    print(f"T peak distance: {np.round(mf_res_t[2], 3)}")

    mf_res_q = matched_filter(0.065, z.q, z.q_mean, z.q_cov, (4, 2))
    print(f"Q peak distance: {np.round(mf_res_q[2], 3)}")

    stats_x = calc_stats(mf_res_x[0], mf_res_x[1])
    stats_y = calc_stats(mf_res_y[0], mf_res_y[1])
    stats_z = calc_stats(mf_res_z[0], mf_res_z[1])
    stats_t = calc_stats(mf_res_t[0], mf_res_t[1])
    stats_q = calc_stats(mf_res_q[0], mf_res_q[1])
    #
    plot_stats(5, [stats_x[0], stats_y[0], stats_z[0], stats_t[0], stats_q[0]],
               [stats_x[1], stats_y[1], stats_z[1], stats_t[1], stats_q[1]],
               [stats_x[2], stats_y[2], stats_z[2], stats_t[2], stats_q[2]],
               [stats_x[3], stats_y[3], stats_z[3], stats_t[3], stats_q[3]],
               [stats_x[4], stats_y[4], stats_z[4], stats_t[4], stats_q[4]],
               ['X', 'Y', 'Z', 'T', 'Q'], 'MF')

    # data = spy.open_image(header)
    # cube = data.load(dtype='double').copy()
    # m8_cube = m8(cube)
    # cov8_cube = cov8(cube, m8_cube)
    # res = matched_filter(0.065, cube, m8_cube, cov8_cube, (4, 2))
    # print(f"peak distance: {np.round(res[2], 3)}")
    # stats = calc_stats(res[0], res[1])
    # plot_stats(1, [stats[0]], [stats[1]], [stats[2]], [stats[3]], [stats[4]])
    # z.save_cubes()
    print('done')
