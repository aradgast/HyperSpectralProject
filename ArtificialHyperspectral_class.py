import numpy as np
from find_nu import find_nu
from m8 import m8
import spectral as spy
import matplotlib.pyplot as plt



class ArtificialHyperspectralCube(spy.SpyFile):
    def __init__(self, header):
        spy.SpyFile.__init__()
        originalData = spy.open_image(header)
        originalData_np = originalData.open_memmap()  # for working with numpy

        self.r, self.c, self.s = originalData.shape
        originalDataPCA = spy.algorithms.principal_components(originalData)
        # eigD = originalDataPCA.eigenvalues  # values are OK but on reverse order
        # eigV = originalDataPCA.eigenvectors  # because eigenvalues on reverse order so as the vectors
        m8originalData = m8(originalData_np)  # it's not a m8 calc, it's a mean for each band
        covOriginalData = originalDataPCA.cov  # values are ok

        ##### Y cube ############

        yCube = originalDataPCA.transform(originalData)  # need to check it
        yCube_np = yCube.image.open_memmap()
        m8yCube = m8(yCube_np)  # much larger values(bigger in 2 order)
        for band in range(self.s):
            yCube_np[:, :, band] *= 1 / np.sqrt(np.var(yCube_np[:, :, band]))

        matrix_y = yCube_np.reshape(self.s, self.r * self.c)
        covYCube = np.cov(matrix_y)

        nuOriginalData = find_nu(originalData, m8originalData, covOriginalData)
        nuYCube = find_nu(yCube, m8yCube, covYCube)

        ##### Z cube ############

        self.data = np.zeros(shape=(self.r, self.c, self.s))

        for band in range(self.s):
            self.data[:, :, band] += np.random.standard_t(nuYCube(band), size=(self.r, self.c))
            self.data[:, :, band] *= 1 / np.sqrt(np.var(z[:, :, band]))

        matrix_z = self.data.reshape(s, r * c)
        self.cov = np.cov(matrix_z)
        self.m8 = m8(z)
        self.nu = find_nu(z, self.m8, self.cov)


if __name__ == "__main__":
    z = ArtificialHyperspectralCube('D1_F12_H2_Cropped_des_Aligned.hdr')
    plt.plot([i for i in range(len(z.nu))], z.nu)