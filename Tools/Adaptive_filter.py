import numpy as np
from scipy.signal import welch
from scipy.ndimage import median_filter
from scipy.stats import kurtosis
from sklearn.preprocessing import RobustScaler, MinMaxScaler


def bandpower_highfreq(x, fs, cutoff_hz):
    X = np.fft.rfft(x - np.mean(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    P = np.abs(X) ** 2
    total = np.sum(P) + 1e-12
    high_idx = freqs >= cutoff_hz
    high = np.sum(P[high_idx])
    return float(high / total)


def spectral_entropy(x, fs, nperseg=None):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)

    freqs, Pxx = welch(x, fs=fs, nperseg=nperseg, detrend='constant')
    Pxx = np.maximum(Pxx, 0.0)

    Psum = np.sum(Pxx) + 1e-12
    Pnorm = Pxx / Psum
    Pnz = Pnorm[Pnorm > 0]
    ent = -np.sum(Pnz * np.log(Pnz))

    Hmax = np.log(len(Pxx)) if len(Pxx) > 0 else 1.0
    return float(ent / (Hmax + 1e-12))


def robust_norm(x):
    rs = RobustScaler()
    nx = rs.fit_transform(x.reshape(-1, 1))
    nx = MinMaxScaler().fit_transform(nx)

    return nx.reshape(x.shape)


def compute_noise_level(data, highfreq_cutoff=1.0):
    fs = 8

    data = np.asarray(data).astype(float)
    num_samples, num_sensors, num_points = data.shape

    highpow = np.zeros((num_samples, num_sensors), dtype=float)
    spec_ent = np.zeros((num_samples, num_sensors), dtype=float)
    kurt = np.zeros((num_samples, num_sensors), dtype=float)

    for i in range(num_samples):
        for j in range(num_sensors):
            x = data[i, j, :]

            highpow[i, j] = bandpower_highfreq(x, fs=fs, cutoff_hz=highfreq_cutoff)

            spec_ent[i, j] = spectral_entropy(x, fs=fs)

            kurt[i, j] = kurtosis(x, fisher=False)

    highpow = robust_norm(highpow)
    spec_ent = robust_norm(spec_ent)
    kurt = robust_norm(kurt)

    noise = (0.6 * spec_ent + 0.3 * highpow + 0.1 * kurt)

    noise_levels = np.clip(noise, 0, 1)

    return noise_levels


def ensure_odd_int(kf):
    k = int(np.round(kf))
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    return k


def map_noise_to_kernel(noise_levels, k_min=3, k_max=27, mode='power'):
    if mode == 'linear':
        kf = k_min + noise_levels * (k_max - k_min)
    elif mode == 'power':
        power = 0.5
        kf = k_min + (noise_levels ** power) * (k_max - k_min)

    vec_ensure = np.vectorize(ensure_odd_int)
    kernels = vec_ensure(kf)

    return kernels


def adaptive_median_filter(data):
    noise_levels = compute_noise_level(data)

    kernels = map_noise_to_kernel(noise_levels)

    num_samples, num_sensors, _ = data.shape
    smoothed_data = np.empty_like(data)

    for i in range(num_samples):
        for j in range(num_sensors):
            smoothed_data[i, j, :] = median_filter(data[i, j, :], size=kernels[i, j], mode='reflect')

    return smoothed_data
