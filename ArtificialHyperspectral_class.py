import numpy as np
from find_nu import find_nu
from m8 import m8
import spectral as spy
import matplotlib.pyplot as plt
from pca import pca
from cov_m8 import cov_m8

class ArtificialHyperspectralCube:
    """ x refers to the original data
    y refers to the transformed data(PCA)"""

    def __init__(self, header):

        self.data = np.ndarray(0)
        original_data = spy.open_image(header)
        self.x_np = np.array(original_data.open_memmap())  # for working with numpy
        self.rowNum, self.colNum, self.bandsNum = original_data.shape
        plt.hist(self.x_np[:, :, 0].flatten(), bins=100)
        plt.show()
        # PCA
        self.y_np, self.cov_x, self.cov_y = pca(self.x_np)
        plt.imshow(self.cov_y)
        plt.colorbar()
        plt.show()
        plt.hist(self.y_np[:, :, 0].flatten(), bins=100)
        plt.xlim(-10, 10)
        plt.show()


        self.m8x = m8(self.x_np)
        self.m8y = m8(self.y_np)

        self.nu_x = find_nu(self.x_np, self.m8x, self.cov_x, False)
        self.nu_y = find_nu(self.y_np, self.m8y, self.cov_y, False)

        # Z cube ############
        self._create_z_cube(self.nu_y)
        self.cov = cov_m8(self.data)
        self.m8 = np.array(m8(self.data))
        self.data -= self.m8
        self.nu = find_nu(self.data, self.m8, self.cov, False)

    def _create_z_cube(self, nu_vec):
        self.data = np.zeros(shape=(self.rowNum, self.colNum, self.bandsNum))
        for band in range(self.bandsNum):
            self.data[:, :, band] += np.random.standard_t(nu_vec[band], size=(self.rowNum, self.colNum))
            self.data[:, :, band] /= self.data[:, :, band].std()

        self.data = np.array(self.data)


if __name__ == "__main__":

    z = ArtificialHyperspectralCube('D1_F12_H2_Cropped_des_Aligned.hdr')
    plt.plot([i for i in range(len(z.nu))], z.nu)
    plt.plot([i for i in range(len(z.m8))], z.m8)
    plt.show()


    pass
