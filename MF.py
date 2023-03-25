#########################################################################################
# All right reserved by BGU, 2023
# Author: Arad Gast, Ido Levokovich
# Date: 03/2023
# Description: this file contains the MF function

#########################################################################################

import numpy as np
import matplotlib.pyplot as plt


def matched_filter(p: float, cube: np.ndarray, m8_cube: np.ndarray, cov: np.ndarray, target: tuple) \
        -> (np.ndarray, np.ndarray):
    """this function implement the MF algorithm on 2 cubes - with and without target.
    params: p, cube, m8, cov, target
    p: the power adding to the target
    cube: the cube to preform on - no subtracting background in assignment(x, not x-m8)
    m8: the m8 calculation to decrease the background effect
    cov: the cube covariance
    target: the index of the wanted target in the cube
    output: matrices after preforming MF - with and without target"""
    target_vec = cube[target[0], target[1]].reshape((1, 1, -1))
    no_target_cube = cube - m8_cube
    inv_cov = np.linalg.inv(cov)

    target_cube = no_target_cube.copy()
    target_cube = target_cube + p * target_vec

    target_mul_inv_phi = np.matmul(target_vec, inv_cov).reshape((1, 1, -1))
    mf_no_target_cube = np.tensordot(no_target_cube, target_mul_inv_phi, axes=([2], [2])).squeeze()
    mf_target_cube = np.tensordot(target_cube, target_mul_inv_phi, axes=([2], [2])).squeeze()

    peak_dist = p * np.tensordot(target_vec, inv_cov, axes=([2], [0])).squeeze()
    peak_dist = np.dot(peak_dist, target_vec.squeeze()).squeeze()

    return mf_target_cube, mf_no_target_cube, peak_dist


if __name__ == "__main__":
    import spectral as spy
    from local_mean_covariance import m8, cov8
    from plot_detection_algo import plot, calc_stats
    import matplotlib.pyplot as plt

    cube = np.random.random(size=(10, 15, 5))
    m8_cube = m8(cube)
    res = matched_filter(1, cube, m8_cube, cov8(cube, m8_cube), (0, 0))
    stats = calc_stats(res[0], res[1], bins=10)
    plot([stats[0]], [stats[1]], [stats[2]], [stats[3]], [stats[4]])
    print("done")
