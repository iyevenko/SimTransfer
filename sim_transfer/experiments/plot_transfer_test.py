#!/home/iyevenko/.pyenv/versions/spinup/bin/python3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

envs = [
    "walker2d_low_friction",
    "walker2d_short_joints",
    "walker2d_long_joints"
]

def plot_transfer(data_dir='../results/transfer_test/', error_bars=False):
    exps = [
        'zeroshot',
        'transfer',
        'scratch'
    ]

    means = []
    stds = []
    for env in envs:
        exp_means = []
        exp_stds = []
        for exp in exps:
            fname = data_dir + env.replace('walker2d', exp)
            t = pd.read_table(fname)
            mean_ret = float(t['AverageEpRet'].values)
            std_ret = float(t['StdEpRet'].values)

            exp_means.append(mean_ret)
            exp_stds.append(std_ret)

        means.append(exp_means)
        stds.append(exp_stds)

    bw = 0.25
    br = np.arange(len(means[0]))

    cs = ['tab:orange', 'tab:blue', 'tab:green']
    labels = [env[len('walker2d_'):] for env in envs]

    for i in range(3):
        x = br+i*bw
        y = means[i]
        yerr = stds[i]
        if error_bars:
            plt.bar(x, y, yerr=yerr, color=cs[i], width=bw, label=labels[i], ecolor='gray', capsize=5)
        else:
            plt.bar(x, y, color=cs[i], width=bw, label=labels[i])

    plt.xticks(br+bw,['Zero-Shot', 'Transfer', 'Scratch'])
    plt.xlabel('Experiment', fontsize=12)
    plt.ylabel('Average Episode Return', fontsize=12)

    plt.title('Cross-Task Transfer Comparison')
    plt.legend()
    plt.show()

plot_transfer()