import numpy as np


def m8(cube):
    rpic, cpic, bands = cube.shape
    m8cube = np.zeros(shape=(rpic,cpic,bands))
    col = np.array([i for i in range(1, cpic)])
    col_p1 = np.array([i+1 for i in range(1, cpic)])
    col_m1 = np.array([i-1 for i in range(1, cpic)])
    row = np.array([i for i in range(1, rpic)])
    row_p1 = np.array([i+1 for i in range(1, rpic)])
    row_m1 = np.array([i-1 for i in range(1, rpic)])
    m8cube[row, col, :] = (cube[row_m1, col_m1, :] + cube[row_m1, col, :] +
                           cube[row_m1, col_p1, :] + cube[row, col_m1, :] + cube[row, col_p1, :] +
                           cube[row_p1, col_m1, :] + cube[row_p1, col, :] + cube[row_p1, col_p1, :]) / 8


    m8cube[0, col, :] = (cube[0, col_m1, :] + cube[1,col_m1, :] + cube[1, col, :] + cube[1, col_p1,:]
                        + cube[rpic, col_p1,:])/5
    m8cube[rpic, col, :] = (cube[rpic, col_m1, :] + cube[rpic-1,col_m1, :] + cube[rpic-1, col, :]
                        + cube[rpic-1, col_p1,:] + cube[rpic, col_p1,:])/5
    m8cube[row, 0, :] = (cube[row_m1, 0, :] + cube[row_m1,1, :] + cube[row, 1, :] + cube[row_p1, 1,:]
                        + cube[row_p1, 0,:])/5
    m8cube[row, cpic, :] = (cube[row_m1, cpic, :] + cube[row_m1,cpic-1, :] + cube[row, cpic-1, :]
                        + cube[row_p1, cpic-1,:] + cube[row_p1, cpic,:])/5

    m8cube[0,0,:] = (cube[0,1,:] + cube[1,1,:] + cube[1,0,:])/3
    m8cube[0,cpic,:] = (cube[0,cpic-1,:] + cube[1, cpic-1, :] + cube[1, cpic, :])/3
    m8cube[rpic,0,:] = (cube[rpic-1,0,:] + cube[rpic-1, 1, :] + cube[rpic, 1, :])/3
    m8cube[rpic,cpic,:] = (cube[rpic-1,cpic-1,:] + cube[rpic-1, cpic, :] + cube[rpic, cpic-1, :])/3

    return m8cube