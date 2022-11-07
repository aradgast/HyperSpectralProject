import numpy as np
from find_nu import find_nu
from m8 import m8
import spectral as spy
import matplotlib.pyplot as plt


class ArtificialHyperspectralCube:
    def __init__(self, header):

        self.data = np.ndarray(0)
        self.original_data = spy.open_image(header)
        original_data_np = self.original_data.open_memmap()  # for working with numpy

        self.rowNum, self.colNum, self.bandsNum = self.original_data.shape
        original_data_pca = spy.algorithms.principal_components(self.original_data)
        # eigD = original_data_pca.eigenvalues  # values are OK but on reverse order
        # eigV = original_data_pca.eigenvectors  # because eigenvalues on reverse order so as the vectors
        self.m8original_data = m8(original_data_np)
        self.cov_original_data = original_data_pca.cov

        # Y cube ############

        self.y_cube = original_data_pca.transform(self.original_data)
        y_cube_np = self.y_cube.image.open_memmap().copy()
        self.m8y_cube = m8(y_cube_np)
        for band in range(self.bandsNum):
            y_cube_np[:, :, band] *= 1 / np.sqrt(np.var(y_cube_np[:, :, band]))

        matrix_y = np.transpose(y_cube_np.reshape(self.rowNum * self.colNum, self.bandsNum))
        self.cov_y_cube = np.cov(matrix_y)

        nu_original_data = find_nu(self.original_data, self.m8original_data, self.cov_original_data)
        nu_y_cube = find_nu(self.y_cube, self.m8y_cube, self.cov_y_cube)

        # Z cube ############
        self.create_z_cube(nu_y_cube)
        self.matrix_z = np.transpose(self.data.reshape(self.rowNum * self.colNum, self.bandsNum))
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
