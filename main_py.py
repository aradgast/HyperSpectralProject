import spectral as spy
import matplotlib.pyplot as plt
import numpy as np
from find_nu import find_nu
from m8 import m8
from ArtificialHyperspectral_class import ArtificialHyperspectralCube

if __name__ == "__main__":
    z = ArtificialHyperspectralCube('D1_F12_H2_Cropped_des_Aligned.hdr')
    plt.figure(1)
    plt.plot([i for i in range(len(z.nu))], z.nu, title='nu')
    plt.figure(2)
    plt.plot([i for i in range(len(z.m8))], z.m8, title='m8')
    plt.figure(3)
    plt.imshow(z.data[:, :, 0].reshape(z.rowNum, z.colNum), cmap='gist_rainbow', title="'0' band")
    plt.show()

    # # Loading the data
    #
    # img = spy.open_image('D1_F12_H2_Cropped_des_Aligned.hdr')
    # img_np = img.open_memmap()  # for working with numpy
    #
    # r, c, s = img.shape
    # pca = spy.algorithms.principal_components(img)
    # eigD = pca.eigenvalues  # values are OK but on reverse order
    # eigV = pca.eigenvectors  # because eigenvalues on reverse order so as the vectors
    # m8x = m8(img_np)  # it's not a m8 calc, it's a mean for each band
    # phi_x = pca.cov  # values are ok
    #
    # # Y cube ############
    #
    # y = pca.transform(img)  # need to check it
    # y_np = y.image.open_memmap().copy()
    # for band in range(s):
    #     y_np[:, :, band] *= 1 / np.sqrt((np.var(y_np[:, :, band])))
    #
    # m8y = m8(y_np)  # much larger values(bigger in 2 order)
    # matrix_y = y_np.reshape(s, r * c)
    # phi_y = np.cov(matrix_y)
    #
    # nu_x = find_nu(img, m8x, phi_x)
    # nu_y = find_nu(y_np, m8y, phi_y)
    #
    # # Z cube ############
    #
    # z = np.zeros(shape=(r, c, s))
    #
    # for band in range(s):
    #     z[:, :, band] += np.random.standard_t(nu_y[band], size=(r, c))
    #     z[:, :, band] *= 1 / np.sqrt(np.var(z[:, :, band]))
    #
    # matrix_z = z.reshape(s, r * c)
    # phi_z = np.cov(matrix_z)
    # m8z = m8(z)
    # nu_z = find_nu(z, m8z, phi_z)
    #
    # plt.imshow(img[:, :, int(input('band:'))].reshape(r, c), cmap='gray')
    # # # plt.imshow(pca_img[:,:,int(input('band:'))].reshape(r,c), cmap='gray')
    # plt.show()
