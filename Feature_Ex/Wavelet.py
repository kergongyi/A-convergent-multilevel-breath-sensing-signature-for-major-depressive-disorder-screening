import numpy as np
import pywt


def get_feature_Wavelet(data):
    num_samples, num_channels, _ = data.shape

    feat_dim = len(wavelet_chn(data[0, 0, :]))
    wvac = np.zeros((num_samples, num_channels * feat_dim))

    for i in range(num_samples):
        ch_features = []
        for ch in range(num_channels):
            feat = wavelet_chn(data[i, ch, :])
            ch_features.append(feat)
        wvac[i, :] = np.concatenate(ch_features)

    return wvac


def wavelet_chn(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    energies = []

    for c in coeffs[1:]:
        energy = np.sum(c ** 2)
        energies.append(energy)

    energies = np.array(energies)
    energy_ratio = energies / (np.sum(energies) + 1e-12)

    cA = coeffs[0]
    features = np.concatenate([energies, energy_ratio, cA])

    features = np.array(features)
    features = (features - np.mean(features)) / (np.std(features) + 1e-12)

    return features
