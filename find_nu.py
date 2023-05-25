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


def find_nu(cube, mean_matrix, cov, method='Constant'):
    '''return the nu vector for the given cube.
    to possible methodas for finding nu:
    1. estimate nu based on james tyler formula
    2. create a cube of t-distribution and find the nu for each band
    '''
    # 1. estimate nu based on james tyler formula
    if method == 'Tyler':
        bands = cube.shape[2]
        nu = np.zeros((bands, 1))
        for i in range(bands):
            r = np.abs(cube[:, :, i] / np.sqrt(cov[i, i]))
            k = np.mean(np.power(r, 3)) / np.mean(r)
            nu[i] = 2 + k / (k - 2)

    # 2. create a cube of t-distribution and find the nu for each band
    elif method == 'KS':
        nu_init = 2  # starts from nu value = 2
        simulation = 200  # number of nu values testing
        comp_mat = np.zeros(shape=(cube.shape[0] * cube.shape[1], simulation))  # save the matrix represent the distribution
        nu_vec = np.zeros(shape=(simulation, 1))
        nu = np.zeros((cube.shape[2], 1))
        statiscis_result = np.zeros((simulation, 1))

        for i in range(simulation):
            comp_mat[:, i] = np.random.standard_t(nu_init, size=cube.shape[0] * cube.shape[1])
            comp_mat[:, i] *= 1 / np.std(comp_mat[:, i])
            comp_mat[:, i] -= np.mean(comp_mat[:, i])
            nu_vec[i] = nu_init
            nu_init += 3 / simulation

        for band in range(cube.shape[2]):
            for sim in range(simulation):
                test = ks_2samp(cube[:, :, band].reshape(cube.shape[0] * cube.shape[1]), comp_mat[:, sim], alternative='two-sided')  # KS test for comparing 2 unknown distribution samples
                statiscis_result[sim] = test[0]
            nu[band] = nu_vec[np.argmin(statiscis_result)]

    elif method == 'Constant':
        nu = np.ones((cube.shape[2], 1)) * 2

    elif method == 'MLE':
        nu = np.zeros((cube.shape[2], 1))
        for band in range(cube.shape[2]):
            stats = t_dist.fit(cube[:, :, band].flatten())
            nu[band] = stats[0]

    elif method == 'NN':
        weights_path = r"C:\Users\gast\PycharmRepos\HyperSpectralProject//best_model3.pt"
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
