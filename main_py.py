import spectral as spy
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from find_nu import find_nu
from local_mean_covariance import m8
from MF import matched_filter
from ArtificialHyperspectral_class import ArtificialHyperspectralCube
from plot_detection_algo import plot

if __name__ == "__main__":
    z = ArtificialHyperspectralCube('D1_F12_H2_Cropped_des_Aligned.hdr')

    z_axis, hist_z_mf_wt, hist_z_mf_nt, z_inv_cumulative_wt,z_inv_cumulative_nt  = matched_filter(0.065, z.data, z.m8, z.cov, (5,3))
    x_axis, hist_x_mf_wt, hist_x_mf_nt, x_inv_cumulative_wt,x_inv_cumulative_nt = matched_filter(0.065, z.x_np, z.m8x, z.cov_x, (5,3))
    y_axis, hist_y_mf_wt, hist_y_mf_nt, y_inv_cumulative_wt,y_inv_cumulative_nt = matched_filter(0.065, z.y_np, z.m8y, z.cov_y, (5,3))

    axis = [z_axis, x_axis, y_axis]
    hist_mf_wt = [hist_z_mf_wt, hist_x_mf_wt, hist_y_mf_wt]
    hist_mf_nt = [hist_z_mf_nt, hist_x_mf_nt, hist_y_mf_nt]
    inv_cumulative_wt = [z_inv_cumulative_wt, x_inv_cumulative_wt, y_inv_cumulative_wt]
    inv_cumulative_nt = [z_inv_cumulative_nt, x_inv_cumulative_nt, y_inv_cumulative_nt]
    legends = ['Z', 'X', 'Y']
    plot(axis, hist_mf_wt, hist_mf_nt, inv_cumulative_wt, inv_cumulative_nt, legends, algo_name='MF')
    print('done')





