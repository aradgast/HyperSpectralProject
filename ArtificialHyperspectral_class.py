import numpy as np
from find_nu import find_nu
from m8 import m8
import spectral as spy
import matplotlib.pyplot as plt


class ArtificialHyperspectralCube:
    def __init__(self, header):

        self.data = np.ndarray(0)
        original_data = spy.open_image(header)
        original_data_np = original_data.open_memmap()  # for working with numpy

        self.rowNum, self.colNum, self.bandsNum = original_data.shape
        original_data_pca = spy.algorithms.principal_components(original_data)
        # eigD = original_data_pca.eigenvalues  # values are OK but on reverse order
        # eigV = original_data_pca.eigenvectors  # because eigenvalues on reverse order so as the vectors
        m8original_data = m8(original_data_np)
        cov_original_data = original_data_pca.cov

        # Y cube ############

        y_cube = original_data_pca.transform(original_data)
        y_cube_np = y_cube.image.open_memmap().copy()
        m8y_cube = m8(y_cube_np)
        for band in range(self.bandsNum):
            y_cube_np[:, :, band] *= 1 / np.sqrt(np.var(y_cube_np[:, :, band]))

        matrix_y = y_cube_np.reshape(self.bandsNum, self.rowNum * self.colNum)
        cov_y_cube = np.cov(matrix_y)

        nu_original_data = find_nu(original_data, m8original_data, cov_original_data)
        nu_y_cube = find_nu(y_cube, m8y_cube, cov_y_cube)

        # Z cube ############
        self.create_z_cube(nu_y_cube)
        self.matrix_z = self.data.reshape(self.bandsNum, self.rowNum * self.colNum)
        self.cov = np.cov(self.matrix_z)
        self.m8 = m8(z)
        self.nu = find_nu(z, self.m8, self.cov)

    def create_z_cube(self, nu_vec):
        self.data = np.zeros(shape=(self.rowNum, self.colNum, self.bandsNum))
        for band in range(self.bandsNum):
            self.data[:, :, band] += np.random.standard_t(nu_vec[band], size=(self.rowNum, self.colNum))
            self.data[:, :, band] *= 1 / np.sqrt(np.var(self.data[:, :, band]))


if __name__ == "__main__":
    z = ArtificialHyperspectralCube('D1_F12_H2_Cropped_des_Aligned.hdr')
    plt.plot([i for i in range(len(z.nu))], z.nu)
    plt.plot([i for i in range(len(z.m8))], z.m8)
    plt.show()
    pass
