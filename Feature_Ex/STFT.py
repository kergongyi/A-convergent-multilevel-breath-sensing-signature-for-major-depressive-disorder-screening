import numpy as np
import scipy.signal as signal
from scipy.stats import entropy


def get_feature_STFT(data):
    num_samples, num_channels, _ = data.shape

    feat_dim = len(stft_chn(data[0, 0, :]))
    stft_f = np.zeros((num_samples, num_channels * feat_dim))

    for i in range(num_samples):
        ch_features = []
        for ch in range(num_channels):
            feat = stft_chn(data[i, ch, :])
            ch_features.extend(list(feat.values()))
        ch_features = np.array(ch_features)
        stft_f[i, :] = ch_features

    return stft_f


def stft_chn(data):
    fs = 8

    nperseg = 64
    noverlap = 32

    f, t, Zxx = signal.stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    S = np.abs(Zxx)

    P = S ** 2
    P_norm = P / (np.sum(P) + 1e-12)

    freq_weight = f[:, None]

    centroid = np.sum(freq_weight * P_norm) / np.sum(P_norm)

    bandwidth = np.sqrt(np.sum(((freq_weight - centroid) ** 2) * P_norm))

    spectral_entropy = entropy(P_norm.flatten())

    flux = np.mean(np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0)))

    mean_power = np.mean(P)

    dom_freq = f[np.argmax(np.sum(P, axis=1))]

    feature_dict = {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_entropy": spectral_entropy,
        "spectral_flux": flux,
        "mean_power": mean_power,
        "dominant_frequency": dom_freq
    }

    return feature_dict

