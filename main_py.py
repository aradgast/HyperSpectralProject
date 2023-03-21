from MF import matched_filter
from ArtificialHyperspectral_class import ArtificialHyperspectralCube
from plot_detection_algo import plot

if __name__ == "__main__":
    z = ArtificialHyperspectralCube('D1_F12_H1_Cropped.hdr')

    z_axis, hist_z_mf_wt, hist_z_mf_nt, z_inv_cumulative_wt, z_inv_cumulative_nt = matched_filter(0.065,
                                                                                                  z.artificial_data,
                                                                                                  z.m8, z.cov, (5, 3))
    x_axis, hist_x_mf_wt, hist_x_mf_nt, x_inv_cumulative_wt, x_inv_cumulative_nt = matched_filter(0.065, z.cube,
                                                                                                  z.x_mean, z.x_cov,
                                                                                                  (5, 3))
    y_axis, hist_y_mf_wt, hist_y_mf_nt, y_inv_cumulative_wt, y_inv_cumulative_nt = matched_filter(0.065, z.y, z.y_mean,
                                                                                                  z.y_cov, (5, 3))
    t_axis, hist_t_mf_wt, hist_t_mf_nt, t_inv_cumulative_wt, t_inv_cumulative_nt = matched_filter(0.065, z.t, z.t_mean,
                                                                                                  z.t_cov, (5, 3))
    q_axis, hist_q_mf_wt, hist_q_mf_nt, q_inv_cumulative_wt, q_inv_cumulative_nt = matched_filter(0.065, z.q, z.q_mean,
                                                                                                  z.q_cov, (5, 3))

    axis = [x_axis, y_axis, z_axis, t_axis, q_axis]
    hist_mf_wt = [hist_x_mf_wt, hist_y_mf_wt, hist_z_mf_wt, hist_t_mf_wt, hist_q_mf_wt]
    hist_mf_nt = [hist_x_mf_nt, hist_y_mf_nt, hist_z_mf_nt, hist_t_mf_nt, hist_q_mf_nt]
    inv_cumulative_wt = [x_inv_cumulative_wt, y_inv_cumulative_wt, z_inv_cumulative_wt, t_inv_cumulative_wt,
                         q_inv_cumulative_wt]
    inv_cumulative_nt = [x_inv_cumulative_nt, y_inv_cumulative_nt, z_inv_cumulative_nt, t_inv_cumulative_nt,
                         q_inv_cumulative_nt]
    legends = ['X', 'Y', 'Z', 'T', 'Q']
    plot(axis, hist_mf_wt, hist_mf_nt, inv_cumulative_wt, inv_cumulative_nt, legends, algo_name='MF')
    print('done')
