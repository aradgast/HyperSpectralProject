#############################################################################################################
# All right reserved by BGU, 2023
# Author: Arad Gast, Ido Levokovich
# Date: 03/2023
# Description: this file contains the plot_detection_algo function

#############################################################################################################

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

def calc_stats(target_cube, no_target_cube, bins=1000):
    """this function calculates the statistics of the detection algorithm
    target_cube - the cube with the target
    no_target_cube - the cube without the target
    bins - the number of bins in the histogram
    """
    max_val = np.max([np.max(target_cube), np.max(no_target_cube)]) * 1.1
    min_val = np.min([np.min(target_cube), np.min(no_target_cube)]) * 1.1
    x = np.linspace(min_val, max_val, bins)

    histogram_wt = np.histogram(target_cube, x)
    histogram_nt = np.histogram(no_target_cube, x)

    # inv_cumulative_probability_wt = np.zeros(shape=(bins, 1))
    # inv_cumulative_probability_nt = np.zeros(shape=(bins, 1))
    # for bin in range(bins):
    #     inv_cumulative_probability_wt[bin] = np.sum(histogram_wt[0][bin:])
    #     inv_cumulative_probability_nt[bin] = np.sum(histogram_nt[0][bin:])
    # inv_cumulative_probability_nt *= 1 / np.max(inv_cumulative_probability_nt)
    # inv_cumulative_probability_wt *= 1 / np.max(inv_cumulative_probability_wt)
    fpr, tpr, thresholds = roc_curve(np.concatenate([np.zeros_like(no_target_cube.flatten()),
                                                     np.ones_like(target_cube.flatten())]),
                                     np.concatenate([no_target_cube.flatten(), target_cube.flatten()]))

    return histogram_wt, histogram_nt, fpr, tpr, thresholds


def plot_stats(number_of_cubes, hist_wt, hist_nt, fpr, tpr, thresholds, legends=['X'], algo_name='MF', name='ViaReggio', method='Constant2'):
    """this function plots the results of the detection algorithm
    axis - the axis of the cumulative probability
    hist_wt - the histogram of the WT
    hist_nt - the histogram of the NT
    inv_cumulative_wt - the inverse cumulative probability of the WT
    inv_cumulative_nt - the inverse cumulative probability of the NT
    legends - the legends of the plots
    algo_name - the name of the algorithm

    returns: None"""

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    title1 = f'{algo_name} - histogram results for '
    title2 = f'{algo_name} - log10 histogram results for '
    title3 = f'{algo_name} - inverse cumulative probability for '
    title4 = f'{algo_name} - ROC curve with limited pfa for '
    colors = ['g', 'b', 'y', 'r', 'k', 'c', 'm', 'w']

    for i in range(number_of_cubes):
        ax[0, 0].plot(hist_wt[i][1][1:], hist_wt[i][0],
                      label=f'{legends[i]}_WT', color=colors[i], linewidth=number_of_cubes - i)
        ax[0, 0].plot(hist_nt[i][1][1:], hist_nt[i][0],
                      '--', label=f'{legends[i]}_NT', color=colors[i], linewidth=number_of_cubes - i)
        title1 += legends[i]
        # ax[0, 0].set_xlim([np.min(hist_wt[i][1][1:]), np.max(hist_wt[i][1][1:])])
        ax[0, 0].set_xlim(-1000, 1000)
        ax[0,0].set_ylabel('Number of samples')
        ax[0,0].set_xlabel('Detection score')
        ax[0, 0].grid()
        ax[0, 0].legend()
        try:
            ax[0, 1].plot(hist_wt[i][1][1:], np.log10(hist_wt[i][0]),
                          label=f'{legends[i]}_WT', color=colors[i], linewidth=number_of_cubes - i)
            ax[0, 1].plot(hist_nt[i][1][1:], np.log10(hist_nt[i][0]),
                          '--', label=f'{legends[i]}_NT', color=colors[i], linewidth=number_of_cubes - i)
            title2 += legends[i]
            # ax[0, 1].set_xlim([np.min(hist_wt[i][1][1:]), np.max(hist_wt[i][1][1:])])
            ax[0, 1].set_xlim(-1000, 1000)
            ax[0, 1].set_ylabel('log10(Number of samples)')
            ax[0, 1].set_xlabel('Detection score')
            ax[0, 1].grid()
            ax[0, 1].legend()
        except Exception as e:
            print(e)

        ax[1, 0].plot(thresholds[i][::-1], tpr[i][::-1],
                      label=f'{legends[i]}_WT', color=colors[i], linewidth=number_of_cubes - i)
        ax[1, 0].plot(thresholds[i][::-1], fpr[i][::-1],
                      '--', label=f'{legends[i]}_NT', color=colors[i], linewidth=number_of_cubes - i)
        title3 += legends[i]
        # ax[1, 0].set_xlim([np.min(hist_wt[i][1][1:]), np.max(hist_wt[i][1][1:])])
        ax[1, 0].set_xlim(-1000, 1000)
        ax[1, 0].set_ylabel('Probability')
        ax[1, 0].set_xlabel('Detection score')
        ax[1, 0].grid()
        ax[1, 0].legend()

        idx = len(fpr[i][fpr[i] <= 0.01])
        roc_auc = auc(fpr[i][:idx], tpr[i][:idx])
        print(f"the AUC for {legends[i]} is {roc_auc}")
        if legends[i] == 'X':
            ax[1, 1].plot(fpr[i], tpr[i],
                      label=f"{legends[i]}: AUC = {np.round(roc_auc,3)}", color=colors[i], linewidth=number_of_cubes - i)
            X_auc = roc_auc
        else:
            realtive_error = (np.abs(roc_auc - X_auc) / X_auc) * 100
            ax[1, 1].plot(fpr[i], tpr[i],
                      label=f"{legends[i]}_rel_error AUC = {np.round(realtive_error,5)} %", color=colors[i], linewidth=number_of_cubes - i)
        ax[1, 1].set_xlabel('False Positive Rate')
        ax[1, 1].set_ylabel('True Positive Rate')
        ax[1, 1].set_xlim([0, 0.01])
        ax[1, 1].grid()
        ax[1, 1].legend()

    ax[0, 0].set_title(title1)
    ax[0, 1].set_title(title2)
    ax[1, 0].set_title(title3)
    ax[1, 1].set_title(title4)
    fig.tight_layout()
    plt.savefig(f'plots/{name}_{algo_name}_{method}_results.png')
    plt.show()


if __name__ == '__main__':
    print('This is a function file, not a main file')
