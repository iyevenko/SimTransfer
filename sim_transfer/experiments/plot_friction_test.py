#!/home/iyevenko/.pyenv/versions/spinup/bin/python3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

envs = [
    "walker2d_friction_80",
    "walker2d_friction_60",
    "walker2d_friction_40",
    "walker2d_friction_20",
]

def plot_friction(data_dir='../results/friction_test/'):
    exps = [
        'zeroshot',
        'transfer',
    ]

    means = {}
    for exp in exps:
        env_means = []
        for env in envs:
            fname = data_dir + env.replace('walker2d', exp)
            t = pd.read_table(fname)
            mean_ret = float(t['AverageEpRet'].values)

            env_means.append(mean_ret)

        means[exp] = env_means

    x = np.arange(len(envs))

    for exp in exps:
        plt.plot(x, means[exp], label=exp)

    ax = plt.gca()
    ax.set_yscale('log')

    plt.xticks(x, [env[-2:] for env in envs])
    plt.xlabel('Environment Friction (%)', fontsize=12)
    plt.ylabel('Average Episode Return', fontsize=12)

    plt.title('Low Friction Transferability Curve')
    plt.legend()
    plt.grid(which='both', linewidth=.2)
    plt.show()


plot_friction()