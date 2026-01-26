import numpy as np


def Single_feature_extractor(data):
    num_samples, num_channels, _ = data.shape

    k = 1

    features = np.zeros((num_samples, num_channels * k))

    for i in range(num_samples):
        for j in range(num_channels):
            feature = peak_sharpness(data[i, j])
            features[i, j * k:(j + 1) * k] = feature

    return features


def TimeConstant_X(data):
    max_val = np.max(data)
    max_idx = np.argmax(data)

    X = 0.6
    threshold = max_val * X
    fs = 8

    indices = np.where(data[:max_idx + 1] >= threshold)[0]
    if len(indices) > 0:
        time = indices[np.argmin(np.abs(data[indices] - threshold))] / fs
    else:
        time = np.nan

    return time


def Dropping_MaxC(data):
    max_idx = np.argmax(data)

    dropping_part = data[max_idx:]

    dy = np.gradient(dropping_part)
    d2y = np.gradient(dy)

    curvature = np.abs(d2y) / (1 + dy ** 2) ** 1.5

    return np.max(curvature)


def Integral_Asymmetry(data):
    fs = 8

    max_idx = np.argmax(data)

    rising_part = data[:max_idx + 1]
    rising_area = np.trapz(np.abs(rising_part), dx=1.0 / fs)

    dropping_part = data[max_idx:]
    dropping_area = np.trapz(np.abs(dropping_part), dx=1.0 / fs)

    if dropping_area > 1e-9:
        asym_index = rising_area / dropping_area
    else:
        asym_index = np.nan

    return asym_index


def peak_sharpness(data):
    fs = 8

    max_v = np.max(data)

    auc = np.trapz(np.abs(data), dx=1.0 / fs)

    if auc <= 1e-12:
        return np.nan

    return max_v / auc
