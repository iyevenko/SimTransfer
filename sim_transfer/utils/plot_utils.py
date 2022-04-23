import numpy as np
import pandas as pd


def smooth_data(x, smooth):
    y = np.ones(smooth)
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    return smoothed_x


def read_data(data_file, n=1, smooth=5):
    t = pd.read_table(data_file)
    epoch = t['Epoch'].values
    ret = t['AverageEpRet'].values
    ret_low = ret-t['StdEpRet'].values
    ret_hi = ret+t['StdEpRet'].values

    if smooth > 1:
        ret = smooth_data(ret, smooth)
        ret_low = smooth_data(ret_low, smooth)
        ret_hi = smooth_data(ret_hi, smooth)
    epoch = epoch[::n]
    ret = ret[::n]
    ret_low = ret_low[::n]
    ret_hi = ret_hi[::n]
    return epoch, ret, ret_low, ret_hi



