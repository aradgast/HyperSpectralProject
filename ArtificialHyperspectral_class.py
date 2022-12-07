import numpy as np
from find_nu import find_nu
from m8 import m8
import spectral as spy
import matplotlib.pyplot as plt


class ArtificialHyperspectralCube:
    """ x refers to the original data
    y refers to the transformed data(PCA)"""

    def __init__(self, header):

        self.data = np.ndarray(0)
        original_data = spy.open_image(header)
        self.x_np = np.array(original_data.open_memmap())  # for working with numpy

        self.rowNum, self.colNum, self.bandsNum = original_data.shape
        original_data_pca = spy.algorithms.principal_components(original_data)
        # eigD = original_data_pca.eigenvalues  # values are OK but on reverse order
        # eigV = original_data_pca.eigenvectors  # because eigenvalues on reverse order so as the vectors
        self.m8x = m8(self.x_np)
        self.cov_x = np.array(original_data_pca.cov)

        # Y cube ############

        y_cube = original_data_pca.transform(original_data)
        self.y_np = np.array(y_cube.image.open_memmap().copy())
        self.m8y = m8(self.y_np)
        for band in range(self.bandsNum):
            self.y_np[:, :, band] *= 1 / np.sqrt(np.var(self.y_np[:, :, band]))

        matrix_y = np.transpose(self.y_np.reshape(self.rowNum * self.colNum, self.bandsNum))
        self.cov_y = np.cov(matrix_y)

        nu_x = find_nu(self.x_np, self.m8x, self.cov_x, False)
        nu_y = find_nu(self.y_np, self.m8y, self.cov_y, True)

        # Z cube ############
        self.create_z_cube(nu_y)
        self.matrix_z = np.transpose(self.data.reshape(self.rowNum * self.colNum, self.bandsNum))
        self.cov = np.cov(self.matrix_z)
        self.m8 = np.array(m8(self.data))
        self.data -= self.m8
        self.nu = find_nu(self.data, self.m8, self.cov, False)

    def create_z_cube(self, nu_vec):
        self.data = np.zeros(shape=(self.rowNum, self.colNum, self.bandsNum))
        for band in range(self.bandsNum):
            self.data[:, :, band] += np.random.standard_t(nu_vec[band], size=(self.rowNum, self.colNum))
            self.data[:, :, band] *= 1 / np.std(self.data[:, :, band])
        self.data = np.array(self.data)


if __name__ == "__main__":

    z = ArtificialHyperspectralCube('D1_F12_H2_Cropped_des_Aligned.hdr')
    plt.plot([i for i in range(len(z.nu))], z.nu)
    plt.plot([i for i in range(len(z.m8))], z.m8)
    plt.show()


    pass
