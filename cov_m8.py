import numpy as np
from m8 import m8

def cov_m8(cube):
    cube -= m8(cube)
    cube = np.reshape(cube, (cube.shape[0]*cube.shape[1], cube.shape[2]))
    return np.matmul(np.transpose(cube), cube)


if __name__ == "__main__":
    # write an easy test for cov_m8 using random data
    a = np.random.rand(10, 15, 5)
    print(cov_m8(a).shape)


