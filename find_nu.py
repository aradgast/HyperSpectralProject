############################################################################################################
# All right reserved by BGU, 2023
# Author: Arad Gast, Ido Levokovich
# Date: 03/2023
# Description: This file contains the find_nu function

#############################################################################################################
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import t as t_dist
from DL_DOF import DOFNet
import torch
import torch.nn as nn
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_nu(cube, mean_matrix, cov, method='Constant2'):
    """return the nu vector for the given cube.
    to possible methods for finding nu:\n
    ['Tyler', 'KS', 'Constant2', 'Constant3', 'MLE', 'NN']
    :param cube: the given cube
    :param mean_matrix: the mean matrix of the given cube
    :param cov: the covariance matrix of the given cube
    :param method: the method for finding nu
    :return: nu vector
    """
    cube_no_mean = np.subtract(cube, mean_matrix)
    # 1. estimate nu based on james tyler formula
    if method == 'Tyler':
        bands = cube.shape[2]
        nu = np.zeros((bands, 1))
        for i in range(bands):
            r = np.abs(cube_no_mean[:, :, i] / np.sqrt(cov[i, i]))
            k = np.mean(np.power(r, 3)) / np.mean(r)
            if k <= 2:
                nu[i] = 0
            else:
                nu[i] = 2 + k / (k - 2)

    # 2. create a cube of t-distribution and find the nu for each band
    elif method == 'KS':
        nu_init = 2  # starts from nu value = 2
        simulation = 200  # number of nu values testing
        comp_mat = np.zeros(
            shape=(cube.shape[0] * cube.shape[1], simulation))  # save the matrix represent the distribution
        nu_vec = np.zeros(shape=(simulation, 1))
        nu = np.zeros((cube.shape[2], 1))
        statiscis_result = np.zeros((simulation, 1))

        for i in range(simulation):
            comp_mat[:, i] = np.random.standard_t(nu_init, size=cube.shape[0] * cube.shape[1])
            comp_mat[:, i] *= 1 / np.std(comp_mat[:, i])
            comp_mat[:, i] -= np.mean(comp_mat[:, i])
            nu_vec[i] = nu_init
            nu_init += 20 / simulation

        for band in range(cube.shape[2]):
            for sim in range(simulation):
                test = ks_2samp(cube_no_mean[:, :, band].reshape(cube.shape[0] * cube.shape[1]), comp_mat[:, sim],
                                alternative='two-sided')  # KS test for comparing 2 unknown distribution samples
                statiscis_result[sim] = test[0]
            nu[band] = nu_vec[np.argmin(statiscis_result)]

    elif method == 'Constant0.5':
        nu = np.ones((cube.shape[2], 1)) * 0.5

    elif method == 'Constant1':
        nu = np.ones((cube.shape[2], 1)) * 1

    elif method == 'Constant1.5':
        nu = np.ones((cube.shape[2], 1)) * 1.5

    elif method == 'Constant2':
        nu = np.ones((cube.shape[2], 1)) * 2

    elif method == 'Constant2.5':
        nu = np.ones((cube.shape[2], 1)) * 2.5

    elif method == 'Constant3':
        nu = np.ones((cube.shape[2], 1)) * 3

    elif method == 'Constant3.5':
        nu = np.ones((cube.shape[2], 1)) * 3.5

    elif method == 'Constant4':
        nu = np.ones((cube.shape[2], 1)) * 4

    elif method == 'Constant4.5':
        nu = np.ones((cube.shape[2], 1)) * 4.5

    elif method == 'Constant5':
        nu = np.ones((cube.shape[2], 1)) * 5

    elif method == 'Constant5.5':
        nu = np.ones((cube.shape[2], 1)) * 5.5

    elif method == 'Constant6':
        nu = np.ones((cube.shape[2], 1)) * 6

    elif method == 'Constant6.5':
        nu = np.ones((cube.shape[2], 1)) * 6.5

    elif method == 'Constant7':
        nu = np.ones((cube.shape[2], 1)) * 7

    elif method == 'Constant7.5':
        nu = np.ones((cube.shape[2], 1)) * 7.5

    elif method == 'Constant8':
        nu = np.ones((cube.shape[2], 1)) * 8

    elif method == 'MLE':
        nu = np.zeros((cube.shape[2], 1))
        for band in range(cube.shape[2]):
            stats = t_dist.fit((cube_no_mean[:, :, band]).flatten())
            nu[band] = stats[0]

    elif method == 'NN':
        weights_path = r"C:\Users\gast\PycharmRepos\HyperSpectralProject\weights//best_model.pt"
        net = DOFNet()
        net.load_state_dict(torch.load(weights_path, map_location=device))
        net.eval()
        net.to(device)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))])
        image = transform(cube).to(device)

        nu = []
        with torch.no_grad():
            for band in range(cube.shape[2]):
                input_band = image[band, :, :].unsqueeze(0)
                output = net(input_band)
                nu.append(output.item())
    else:
        raise ValueError('method not found')

    return nu


if __name__ == "__main__":
    # weights_path = r"C:\Users\gast\PycharmRepos\HyperSpectralProject\weights\best_model.pt"
    # net = DOFNet()
    # net.load_state_dict(torch.load(weights_path, map_location=device))
    # print(net)
    # net.eval()
    # net.to(device)
    # import spectral as spy
    # data = spy.open_image('self_test_rad.hdr')
    # # convert the data to a numpy array
    # data = np.array(data.open_memmap())
    # data = data[:, :, 0:5].astype(np.float32)
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((224, 224))])
    # data = transform(data).to(device)
    # output = net(data)
    # print(output)
    pass