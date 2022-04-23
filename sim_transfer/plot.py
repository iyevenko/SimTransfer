#!/home/iyevenko/.pyenv/versions/spinup/bin/python3
import sys

import matplotlib.pyplot as plt

from utils.plot_utils import read_data


exps = [
    "walker2d_base",          # 0
    "walker2d_low_friction",  # 1
    "walker2d_short_joints",  # 2
    "walker2d_long_joints",   # 3
    "transfer_low_friction",  # 4
    "transfer_short_joints",  # 5
    "transfer_long_joints",   # 6
    "transfer_friction_80",   # 7
    "transfer_friction_60",   # 8
    "transfer_friction_40",   # 9
    "transfer_friction_20",   # 10
]
exp_in = sys.argv[1]
if exp_in in exps:
    exp_name = exp_in
else:
    try:
        exp_ind = int(exp_in)
        exp_name = exps[exp_ind]
    except ValueError:
        print("ERROR: Experiment index must be an integer")
        raise
    except IndexError:
        print('ERROR: Must provide index from 0-6')
        raise

f = f'/home/iyevenko/Documents/spinningup/data/{exp_name}/{exp_name}_s0/progress.txt'

x, y, y_low, y_hi = read_data(f, n=1, smooth=1)

plt.plot(x, y, color='b')
plt.fill_between(x, y_low, y_hi, color='b', alpha=0.2)
plt.show()