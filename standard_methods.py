import numpy as np
from numpy import ndarray
from scipy.signal import spectrogram
from constants import fs

def get_most_common_freq(signal: ndarray, **kwargs) -> ndarray:
    nperseg = kwargs['nperseg'] if 'nperseg' in kwargs.keys() else 512
    noverlap = kwargs['noverlap'] if 'noverlap' in kwargs.keys() else 256

    f, t, Sxx = spectrogram(signal, fs, nperseg=nperseg, noverlap=noverlap)

    freq_common = np.mean(Sxx, axis=1)
    return freq_common, f


if __name__ == "__main__":
    pass