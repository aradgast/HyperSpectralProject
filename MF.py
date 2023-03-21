import numpy as np
import matplotlib.pyplot as plt


def matched_filter(p: float, cube: np.ndarray, m8: np.ndarray, cov: np.ndarray, target: tuple) \
        -> (np.ndarray, np.ndarray):
    """this function implement the MF algorithm on 2 cubes - with and without target.
    params: p, cube, m8, cov, target
    p: the power adding to the target
    cube: the cube to preform on - no subtracting background in assignment(x, not x-m8)
    m8: the m8 calculation to decrease the background effect
    cov: the cube covariance
    target: the index of the wanted target in the cube
    output: matrices after preforming MF - with and without target"""
    row_num, col_num, band_num = cube.shape
    target_vec = cube[target[0], target[1]]
    no_target_cube = cube - m8
    inv_cov = np.linalg.inv(cov)

    target_cube = np.zeros(shape=(row_num, col_num, band_num))
    for row in range(row_num):
        for col in range(col_num):
            target_cube[row, col, :] += target_vec * p
    target_cube += no_target_cube

    target_mul_inv_phi = np.matmul(target_vec, inv_cov)
    mt_no_target_cube = np.zeros(shape=(row_num, col_num))
    mf_target_cube = np.zeros(shape=(row_num, col_num))
    for row in range(row_num):
        for col in range(col_num):
            mt_no_target_cube[row, col] = np.matmul(target_mul_inv_phi, no_target_cube[row, col, :])
            mf_target_cube[row, col] = np.matmul(target_mul_inv_phi, target_cube[row, col, :])

    bins = 1000
    x = np.linspace(-1000, 1000, bins)

    plot_mf_wt = np.histogram(mf_target_cube, x)
    plot_mf_nt = np.histogram(mt_no_target_cube, x)

    inv_cumulative_probability_wt = np.zeros(shape=(bins, 1))
    inv_cumulative_probability_nt = np.zeros(shape=(bins, 1))
    for bin in range(bins):
        inv_cumulative_probability_wt[bin] = np.sum(plot_mf_wt[0][bin:])
        inv_cumulative_probability_nt[bin] = np.sum(plot_mf_nt[0][bin:])
    inv_cumulative_probability_nt *= 1 / np.max(inv_cumulative_probability_nt)
    inv_cumulative_probability_wt *= 1 / np.max(inv_cumulative_probability_wt)

    return x, plot_mf_wt, plot_mf_nt, inv_cumulative_probability_wt, inv_cumulative_probability_nt


if __name__ == "__main__":
    import spectral as spy
    from local_mean_covariance import m8

    # Example
    img = spy.open_image('D1_F12_H2_Cropped_des_Aligned.hdr')
    img_np = np.array(img.open_memmap().copy())  # for working with numpy
    phi = np.cov(np.transpose(img_np.reshape(img_np.shape[0] * img_np.shape[1], img_np.shape[2])))
    no_target_x, target_y = matched_filter(0.065, img_np, m8(img_np), phi, (5, 3), True)
    plt.show()
    pass
