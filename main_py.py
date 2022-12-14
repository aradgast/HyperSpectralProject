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
    plt.show()

    plt.figure(2)
    plt.plot(x_axis, x_inv_cumulative_wt, x_axis, x_inv_cumulative_nt, '--')
    plt.plot(y_axis, y_inv_cumulative_wt, y_axis, y_inv_cumulative_nt, '--')
    plt.plot(z_axis, z_inv_cumulative_wt, z_axis, z_inv_cumulative_nt, '--')
    plt.title('MF - inverse cumulative probability for X&Y&Z')
    plt.grid()
    plt.legend(['WT - X', 'NT - X', 'WT - Y', 'NT - Y', 'WT - Z', 'NT - Z'])
    plt.show()

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
    plt.show()

    pass    #for debugging




