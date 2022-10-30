import spectral as spy
import matplotlib.pyplot as plt
import numpy as np

img = spy.open_image('D1_F12_H2_Cropped_des_Aligned.hdr')
pca_img = spy.principal_components(img)
r, c, s = img.shape
plt.imshow(img[:,:,int(input('band:'))].reshape(r,c), cmap='gray')
# plt.imshow(pca_img[:,:,int(input('band:'))].reshape(r,c), cmap='gray')
plt.show()
