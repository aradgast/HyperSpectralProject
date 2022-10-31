import spectral as spy
import matplotlib.pyplot as plt
import numpy as np
from find_nu import find_nu
from m8 import m8


img = spy.open_image('D1_F12_H2_Cropped_des_Aligned.hdr')
r, c, s = img.shape

pca = spy.algorithms.principal_components(img)
eigD = pca.eigenvalues  # values are OK
eigV = pca.eigenvectors  # values not the same, size is ok
m8x = m8(img)  # it's not a m8 calc, it's a mean for each band
phi_x = pca.cov  # values are ok

y = pca.transform(img)  # need to check it
pca_y = spy.algorithms.principal_components(y)  # same
m8y = m8(y)  # same
phi_y = pca_y.cov  # we get the same cov as x

nu_x = find_nu(img, m8x, phi_x)
nu_y = find_nu(y, m8y, phi_y)

plt.imshow(img[:, :, int(input('band:'))].reshape(r, c), cmap='gray')
# plt.imshow(pca_img[:,:,int(input('band:'))].reshape(r,c), cmap='gray')
plt.show()
