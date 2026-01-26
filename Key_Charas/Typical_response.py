import numpy as np
import scipy.io as sio

from dtaidistance import dtw
from dtaidistance.dtw_barycenter import dba

import Tools.Outlier_identification


def dtw_barycenter_chn(channel_data):
    initial_center = channel_data[0]

    center_waveform = dba(channel_data, c=initial_center)

    return center_waveform


def dtw_distance_to_center_chn(channel_data, center):
    num_samples = channel_data.shape[0]
    distances = np.zeros(num_samples)

    for i in range(num_samples):
        distances[i] = dtw.distance_fast(channel_data[i], center)

    return distances


def fusion_score_chn(channel_data, center, dtw_distances):
    metric_dtw = dtw_distances

    energy_samples = np.sum(channel_data ** 2, axis=1)
    energy_center = np.sum(center ** 2)
    metric_energy = np.abs(energy_samples - energy_center) / energy_center

    diff = channel_data - center
    metric_residual = np.linalg.norm(diff, axis=1) / np.linalg.norm(center)

    all_metrics = np.vstack([metric_dtw, metric_energy, metric_residual])
    all_metrics_z = (all_metrics - np.mean(all_metrics, axis=1, keepdims=True)) / np.std(all_metrics, axis=1,
                                                                                         keepdims=True)
    weights = np.array([0.8, 0.1, 0.1])
    weights = np.array(weights) / np.sum(weights)

    fusion_score = np.dot(weights, all_metrics_z)

    return fusion_score


# Assume H_D denotes the healthy control group
# D_D denotes the MDD group

num_channels = H_D.shape[1]
num_samples = H_D.shape[0]

fusion_scores = np.zeros((num_channels, num_samples))

for chn in range(num_channels):
    print(f"Processing channel {chn + 1}/{num_channels}...")

    channel_data = H_D[:, chn, :]

    clean_mask = Tools.Outlier_identification.detect_shape_outliers_chn(channel_data, dtw_thresh=2.0)

    center = dtw_barycenter_chn(channel_data[clean_mask])

    distances = dtw_distance_to_center_chn(channel_data, center)

    scores = fusion_score_chn(channel_data, center, distances)

    fusion_scores[chn, :] = scores

fusion_scores_mean = np.mean(fusion_scores, axis=0)

best_indices = np.argsort(fusion_scores_mean)[:1]

typical_samples = H_D[best_indices, :, :]
sio.savemat('typical_samples_Health.mat', {
    'representative_samples': typical_samples
})
