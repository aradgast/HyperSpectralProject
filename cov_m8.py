import numpy as np

def cov_m8(cube, m8):
    cube -= m8
    cube = np.reshape(cube, shape=(cube.shape[0]*cube.shape[1], cube.shape[2]))
    return np.matmul(np.transpose(cube), cube)