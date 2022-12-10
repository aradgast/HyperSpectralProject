'''this function will preform PCA on the cube and return the transformed cube'''
import numpy as np
from m8 import m8


def pca(cube):
    # get the shape of the cube
    row, col, bands = cube.shape
    # reshape the cube to a 2D matrix
    cube = cube.reshape(row * col, bands)
    # center the data
    cube -= m8(cube)
    # calculate the covariance matrix
    cov = np.cov(cube, rowvar=False)
    # calculate the eigenvalues and eigenvectors
    eigval, eigvec = np.linalg.eig(cov)
    # sort the eigenvalues and eigenvectors
    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    eigvec = np.multiply(eigvec, np.sqrt(eigval) ** (-1))
    # project the data
    cube = np.dot(eigvec.T, cube.T).T
    # update the covariance matrix
    cov_new = np.cov(cube, rowvar=False)
    # reshape the data back to the original shape
    cube = cube.reshape(row, col, bands)

    return cube, cov, cov_new


if __name__ == "__main__":
    import spectral as spy
    import matplotlib.pyplot as plt

    # load the data
    data = spy.open_image('D1_F12_H2_Cropped_des_Aligned.hdr')
    # convert the data to a numpy array
    data = np.array(data.open_memmap())
    # perform PCA
    cube, cov, cov2 = pca(data)
    # plot the data
    plt.imshow(cube[:, :, 0])
    plt.show()
    plt.imshow(cov)
    plt.show()
    plt.imshow(cov2)
    plt.show()
