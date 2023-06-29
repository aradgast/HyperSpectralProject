#############################################################################################################
# All right reserved by BGU, 2023
# Author: Arad Gast, Ido Levokovich
# Date: 03/2023
# Description: this file contains the plot_detection_algo function

#############################################################################################################

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import datetime


def calc_stats(target_cube, no_target_cube, bins=1000):
    """this function calculates the statistics of the detection algorithm
    target_cube - the cube with the target
    no_target_cube - the cube without the target
    bins - the number of bins in the histogram

    returns: the histogram of the WT+NT, the false positive rate, the true positive rate, the thresholds
    """
    bins_value = 1000
    x = np.linspace(-bins_value, bins_value, bins)

    histogram_wt = np.histogram(target_cube, x)
    histogram_nt = np.histogram(no_target_cube, x)

    fpr, tpr, thresholds = roc_curve(np.concatenate([np.zeros_like(no_target_cube.flatten()),
                                                     np.ones_like(target_cube.flatten())]),
                                     np.concatenate([no_target_cube.flatten(), target_cube.flatten()]))

    return histogram_wt, histogram_nt, fpr, tpr, thresholds


def plot_stats(hist_wt, hist_nt, fpr, tpr, thresholds,
               legends=None, algo_name='MF', name_of_the_dataset=None,
               name_of_estimation_method=None, save_fig=True):
    """this function plots the results of the detection algorithm
    axis - the axis of the cumulative probability
    hist_wt - the histogram of the WT
    hist_nt - the histogram of the NT
    inv_cumulative_wt - the inverse cumulative probability of the WT
    inv_cumulative_nt - the inverse cumulative probability of the NT
    legends - the legends of the plots
    algo_name - the name of the algorithm

    returns: None"""
    number_of_cubes = len(hist_wt)
    if len(hist_wt) != len(hist_nt):
        raise ValueError('hist_wt and hist_nt must have the same length')
    if len(fpr) != len(tpr):
        raise ValueError('fpr and tpr must have the same length')
    if legends is None:
        print('legends is None, using default legends')
        legends = ['Cube ' + str(i) for i in range(number_of_cubes)]
    if name_of_estimation_method is None:
        print('name_of_estimation_method is None, using default name')
        name_of_estimation_method = "Generic"
    if name_of_the_dataset is None:
        print('name_of_the_dataset is None, using default name')
        name_of_the_dataset = "Generic"

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    title1 = f'histogram results'
    title2 = f'log10 histogram results'
    title3 = f'inverse cumulative probability'
    title4 = f'ROC curve with limited pfa'
    colors = ['g', 'b', 'r', 'k', 'c', 'm', 'y', 'w']

    for i in range(number_of_cubes):
        ax[0, 0].plot(hist_wt[i][1][1:], hist_wt[i][0],
                      label=f'{legends[i]}_WT', color=colors[i], linewidth=number_of_cubes - i)
        ax[0, 0].plot(hist_nt[i][1][1:], hist_nt[i][0],
                      '--', label=f'{legends[i]}_NT', color=colors[i], linewidth=number_of_cubes - i)
        ax[0, 0].set_xlim(-1000, 1000)
        ax[0, 0].set_ylabel('Number of samples')
        ax[0, 0].set_xlabel('Detection score')
        ax[0, 0].grid()
        ax[0, 0].legend(loc='upper left')
        try:
            ax[0, 1].plot(hist_wt[i][1][1:], np.log10(hist_wt[i][0]),
                          label=f'{legends[i]}_WT', color=colors[i], linewidth=number_of_cubes - i)
            ax[0, 1].plot(hist_nt[i][1][1:], np.log10(hist_nt[i][0]),
                          '--', label=f'{legends[i]}_NT', color=colors[i], linewidth=number_of_cubes - i)
            # ax[0, 1].set_xlim([np.min(hist_wt[i][1][1:]), np.max(hist_wt[i][1][1:])])
            ax[0, 1].set_xlim(-1000, 1000)
            ax[0, 1].set_ylabel('log10(Number of samples)')
            ax[0, 1].set_xlabel('Detection score')
            ax[0, 1].grid()
            ax[0, 1].legend(loc='upper left')
        except Exception as e:
            print(e)

        ax[1, 0].plot(thresholds[i][::-1], tpr[i][::-1],
                      label=f'{legends[i]}_WT', color=colors[i], linewidth=number_of_cubes - i)
        ax[1, 0].plot(thresholds[i][::-1], fpr[i][::-1],
                      '--', label=f'{legends[i]}_NT', color=colors[i], linewidth=number_of_cubes - i)
        # ax[1, 0].set_xlim([np.min(hist_wt[i][1][1:]), np.max(hist_wt[i][1][1:])])
        ax[1, 0].set_xlim(-1000, 1000)
        ax[1, 0].set_ylabel('Probability')
        ax[1, 0].set_xlabel('Detection score')
        ax[1, 0].grid()
        ax[1, 0].legend(loc='lower left')

        idx = len(fpr[i][fpr[i] <= 0.01])
        roc_auc = auc(fpr[i][:idx], tpr[i][:idx])
        print(f"the AUC for {legends[i]} is {roc_auc}")
        if i == 0:
            ax[1, 1].plot(fpr[i], tpr[i],
                          label=f"{legends[i]}: AUC = {np.round(roc_auc, 3)}", color=colors[i],
                          linewidth=number_of_cubes - i)
            X_auc = roc_auc
        else:
            realtive_error = (np.abs(roc_auc - X_auc) / X_auc) * 100
            ax[1, 1].plot(fpr[i], tpr[i],
                          label=f"{legends[i]}_rel_error AUC = {np.round(realtive_error, 5)} %", color=colors[i],
                          linewidth=number_of_cubes - i)
        ax[1, 1].set_xlabel('False Positive Rate')
        ax[1, 1].set_ylabel('True Positive Rate')
        ax[1, 1].set_xlim([0, 0.01])
        ax[1, 1].grid()
        ax[1, 1].legend(loc='lower right')

    ax[0, 0].set_title(title1)
    ax[0, 1].set_title(title2)
    ax[1, 0].set_title(title3)
    ax[1, 1].set_title(title4)
    if save_fig:
        plt.savefig(f"plots/{name_of_the_dataset}_{name_of_estimation_method}_{datetime.datetime.now().strftime('%d/%m/%Y_%H:%M:%S')}.png")
    fig.suptitle(f'{algo_name} performance on {name_of_the_dataset} dataset with {name_of_estimation_method} as DOF estimation', fontsize=10)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    print('This is a function file, not a main file')
