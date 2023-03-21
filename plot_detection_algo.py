import matplotlib.pyplot as plt
import numpy as np


def plot(axis, hist_wt, hist_nt, inv_cumulative_wt, inv_cumulative_nt, legends, algo_name):
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
    title1 = f'{algo_name} - histogram results for'
    title2 = f'{algo_name} - log10 histogram results for'
    title3 = f'{algo_name} - inverse cumulative probability for'
    title4 = f'{algo_name} - ROC curve with limited pfa for'

    for i in range(len(axis)):
        ax[0, 0].plot(hist_wt[i][1][1:], hist_wt[i][0], hist_nt[i][1][1:], hist_nt[i][0], '--',
                      label=[f'{legends[i]}+WT', f'{legends[i]}_NT'])
        title1 += legends[i]
        ax[0, 0].grid()
        ax[0, 0].legend()

        ax[0, 1].plot(np.log10(hist_wt[i][1][1:]), np.log10(hist_wt[i][0]), np.log10(hist_nt[i][1][1:]),
                      np.log10(hist_nt[i][0]), '--', label=[f'{legends[i]}+WT', f'{legends[i]}_NT'])
        title2 += legends[i]
        ax[0, 1].grid()
        ax[0, 1].legend()

        ax[1, 0].plot(axis[i], inv_cumulative_wt[i], axis[i], inv_cumulative_nt[i], '--',
                      label=[f'{legends[i]}+WT', f'{legends[i]}_NT'])
        title3 += legends[i]
        ax[1, 0].grid()
        ax[1, 0].legend()

        ax[1, 1].plot(inv_cumulative_nt[i], inv_cumulative_wt[i], label=[f'{legends[i]}+WT', f'{legends[i]}_NT'])
        title4 += legends[i]
        ax[1, 0].grid()
        ax[1, 0].legend()

    ax[0, 0].set_title(title1)
    ax[0, 1].set_title(title2)
    ax[1, 0].set_title(title3)
    ax[1, 1].set_title(title4)
    fig.tight_layout()
    plt.show()
