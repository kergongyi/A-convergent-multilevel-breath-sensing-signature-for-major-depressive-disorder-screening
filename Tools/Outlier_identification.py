import numpy as np
from dtaidistance import dtw


def detect_shape_outliers_chn(channel_data, dtw_thresh=2.0):
    num_samples = channel_data.shape[0]
    mean_dtw = np.zeros(num_samples)

    for i in range(num_samples):
        dists = []
        for j in range(num_samples):
            if i != j:
                dists.append(dtw.distance_fast(channel_data[i], channel_data[j]))
        mean_dtw[i] = np.mean(dists)

    z_dtw = (mean_dtw - np.mean(mean_dtw)) / np.std(mean_dtw)
    mask = z_dtw < dtw_thresh

    return mask
