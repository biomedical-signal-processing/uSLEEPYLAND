"""
This module contains functions to compute the spectrogram of the PSG data.
"""

import numpy as np
import scipy.signal as sig

def compute_spectrogram(
    X,
    signal_fs=128,
    resample_fs=100,
    Nfir=100,
    f_cutoff=[0.3, 40],
    window="hamming",
    pass_zero="bandpass",
    scale=True,
    nfft=256,
    log_magnitude=True,
    eps=0.0001
):
    """
    Compute the spectrogram of the batched PSG data. If batch size is missing
    (i.e. X has shape [n_epochs, epoch_len, n_channels]), prior to calling this
    function, add a batch dimension to the data, and squeeze it back after if needed.

    Args:
        X: (ndarray)  A batch of data of shape [batch_size, n_epochs, epoch_len, n_channels]
        signal_fs: (int) The sampling frequency of the PSG data
        resample_fs: (int) The resampling frequency of the PSG data
        Nfir: (int) The number of FIR filter taps
        f_cutoff: (list) The cutoff frequencies for the bandpass filter
        window: (str) The window function for the FIR filter
        pass_zero: (str) The type of filter
        scale: (bool) Scale the filter coefficients
        nfft: (int) The number of FFT points
        log_magnitude: (bool) Compute the log magnitude of the spectrogram
        eps: (float) Small number to avoid log(0)

    Returns:
        The spectrogram of the PSG data of shape [batch_size, n_epochs, n_time, n_freq, n_channels]
    """

    assert (len(X.shape) == 4)

    epoch_len = 30 * signal_fs

    X = sig.resample(
        X,
        int(epoch_len / signal_fs * resample_fs),
        axis=2)

    # bandpass filter
    b = sig.firwin(
        numtaps=Nfir + 1,
        cutoff=f_cutoff,
        window=window,
        pass_zero=pass_zero,
        scale=scale,
        fs=resample_fs)
    X = sig.filtfilt(b, 1, X, axis=2)

    # compute spectrogram
    f, t, S_epoch = sig.spectrogram(
        X,
        fs=resample_fs,
        window=window,
        nperseg=2 * resample_fs,
        noverlap=resample_fs,
        nfft=nfft,
        detrend=False,
        scaling='density',
        mode='complex',
        axis=2)

    if log_magnitude:
        S_epoch = 20 * np.log10(np.abs(S_epoch) + eps)

    # transpose to [batch_size, n_epochs, n_time, n_freq, n_channels]
    S_epoch = S_epoch.transpose(0, 1, 4, 2, 3)

    return S_epoch