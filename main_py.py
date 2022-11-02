import spectral as spy
from scipy import stats
#import scipy.stats.kstest as ks
import matplotlib.pyplot as plt
import numpy as np
from find_nu import find_nu
from m8 import m8
from ArtificialHyperspectral_class import ArtificialHyperspectralCube

if __name__ == "__main__":
    # z = ArtificialHyperspectralCube('D1_F12_H2_Cropped_des_Aligned.hdr')
    # plt.figure(1)
    # plt.plot([i for i in range(len(z.nu))], z.nu, title='nu')
    # plt.figure(2)
    # plt.plot([i for i in range(len(z.m8))], z.m8, title='m8')
    # plt.figure(3)
    # plt.imshow(z.data[:, :, 0].reshape(z.rowNum, z.colNum), cmap='gist_rainbow', title="'0' band")
    # plt.show()
    # Loading the data
    #

    ############################################################### ido's attempt
    img = spy.open_image('D1_F12_H2_Cropped_des_Aligned.hdr')
    img_np = img.open_memmap()  # for working with numpy

    rowNum, colNum, bandsNum = img.shape
    print(img[:, :,200].shape)


    # plt.plot(img_np[:, :, 200].reshape(rowNum, colNum))
    plt.imshow(img[:, :,200].reshape(rowNum, colNum))
    plt.figure(1)
    plt.show()

    print(img[:, :,200].shape)

    m8x_cube = m8(img_np)
    matrix_x = img_np.reshape(bandsNum, rowNum*colNum)    # how do I take only the 200 band?
    #matrix_x = matrix_x[200, :]

    print(matrix_x.shape)
    cov_x_cube = np.cov(matrix_x)
    nu_x_cube = find_nu(matrix_x, m8x_cube, cov_x_cube)     # need to use the 200 band
    ArtificialHyperspectralCube.create_z_cube(nu_x_cube)  # how do I take the data from z.data?

    comp_mat = np.zeros(shape=(rowNum, colNum))
    print(comp_mat.shape)

    comp_mat_t = np.random.standard_t(comp_mat, size=(rowNum, colNum)) ##
    stats.kstest(img_np[:, :,200].reshape(rowNum, colNum), comp_mat_t)   # consider using ks_2samp instead

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
