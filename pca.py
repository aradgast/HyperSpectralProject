import numpy as np


def pca(data, mean, cov):
    """ calculate the PCA of the data cube"""

    # get the shape of the cube
    row, col, bands = data.shape
    data -= mean

    eigval, eigvec = np.linalg.eig(cov)
    # sort the eigenvalues and eigenvectors
    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    # project the data
    cube = np.dot(eigvec.T, data.reshape((row * col, bands)).T).reshape((row, col, bands))
    # update the covariance matrix

    return cube


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
    plt.colorbar()
    plt.show()

    pass
