import numpy as np
from vmdpy import VMD
from scipy.stats import entropy
from tqdm import tqdm


def get_feature_vmd(data):
    chnNum = data.shape[1]

    ft_chn_len = len(vmd_chn(data[0][0]))
    vmd_ft = np.zeros((len(data), ft_chn_len * chnNum))

    for i in tqdm(range(len(data)), desc='Feature extracting'):
        for j in range(chnNum):
            vmd_ft[i, j * ft_chn_len: (j + 1) * ft_chn_len] = vmd_chn(data[i][j])

    return vmd_ft


def permutation_entropy(time_series, order=3, delay=1):
    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))
    sorted_idx = np.array([np.argsort(x[i:i + order * delay:delay]) for i in range(len(x) - (order - 1) * delay)]).T
    hashval = (np.multiply(sorted_idx.T, hashmult)).sum(1)
    _, c = np.unique(hashval, return_counts=True)
    p = c / c.sum()
    return -np.sum(p * np.log2(p))


def vmd_chn(data):
    alpha = 2000
    tau = 0
    K = 4
    DC = 0
    init = 1
    tol = 1e-7

    u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)

    features = []

    imf_energies = np.sum(u ** 2, axis=1)
    total_energy = np.sum(imf_energies)

    for i, imf in enumerate(u):
        abs_imf = np.abs(imf)
        rms_val = np.sqrt(np.mean(imf ** 2))

        shape_factor = rms_val / (np.mean(abs_imf) + 1e-10)

        pe_val = permutation_entropy(imf, order=5, delay=1)

        energy_ratio = imf_energies[i] / (total_energy + 1e-10)

        imf_feats = [
            rms_val,
            shape_factor,
            pe_val,
            energy_ratio
        ]
        features.extend(imf_feats)

    energy_dist_entropy = entropy(imf_energies)
    features.append(energy_dist_entropy)

    return np.array(features)
