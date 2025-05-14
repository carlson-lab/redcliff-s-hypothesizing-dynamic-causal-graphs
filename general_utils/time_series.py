import numpy as np
import pywt
import random
from scipy.signal import butter, lfilter, iirnotch, csd
from .directed_spectrum import get_directed_spectrum


# WAVELET FUNCTIONS/METHODS ##################################################################

def perform_wavelet_decomposition(orig_sig, wavelet_type, level, decomposition_type):
    assert len(orig_sig.shape) == 3 # assumed to be of shape (1, time_steps, n_chans)
    wavelet = pywt.Wavelet(wavelet_type)
    orig_sig = orig_sig[0,:,:].T
    decomp_sig = np.zeros((orig_sig.shape[0]*(level+1), orig_sig.shape[1]))

    for c in range(len(orig_sig)):
        if decomposition_type == "wavedec":
            orig_sig[c,:] = pywt.wavedec(orig_sig[c,:], wavelet, level=level)
        elif decomposition_type == "swt":
            decomp = pywt.swt(orig_sig[c,:], wavelet, level=level, trim_approx=True, norm=True)
            for i, d in enumerate(decomp):
                decomp_sig[c*(level+1)+i,:] += d
        else:
            raise NotImplementedError("general_utils.misc_utils.perform_wavelet_decomposition: Unrecognized decomposition_type == "+str(decomposition_type))
    
    return np.expand_dims(decomp_sig.T, axis=0)


def construct_signal_approx_from_wavelet_coeffs(coeffs, level, wavelet_coeff_type="additive"):
    assert len(coeffs.shape) == 3
    assert coeffs.shape[0] == 1
    curr_channel = 0

    if wavelet_coeff_type == "additive":
        for i in range(level+1):
            if i==0:
                approx = coeffs[0,:,[j for j in range(coeffs.shape[-1]) if j%(level+1)==0]]
            else:
                approx += coeffs[0,:,[j for j in range(coeffs.shape[-1]) if j%(level+1)==i]]
        return approx

    raise NotImplementedError("general_utils.misc_utils.construct_signal_approx_from_wavelet_coeffs: Unrecognized wavelet_coeff_type == "+str(wavelet_coeff_type))
    pass


# # TEMPORAL FEATURE GENERATION ##################################################################
# """
# Compute power-spectral and related features for various time series
# References:
#  - https://github.com/carlson-lab/lpne/blob/master/lpne/preprocess/make_features.py (and relevant dependencies)
# """

def unsqueeze_triangular_array(arr, dim=0):
    """
    Transform a numpy array from condensed triangular form to symmetric form.

    Parameters
    ----------
    arr : numpy.ndarray
    dim : int
        Axis to expand

    Returns
    -------
    new_arr : numpy.ndarray
        Expanded array
    """
    n = int(round((-1 + np.sqrt(1 + 8 * arr.shape[dim])) / 2))
    assert (n * (n + 1)) // 2 == arr.shape[
        dim
    ], f"{(n * (n+1)) // 2} != {arr.shape[dim]}"
    arr = np.swapaxes(arr, dim, -1)
    new_shape = arr.shape[:-1] + (n, n)
    new_arr = np.zeros(new_shape, dtype=arr.dtype)
    for i in range(n):
        for j in range(i + 1):
            idx = (i * (i + 1)) // 2 + j
            new_arr[..., i, j] = arr[..., idx]
            if i != j:
                new_arr[..., j, i] = arr[..., idx]
    dim_list = list(range(new_arr.ndim - 2)) + [dim]
    dim_list = dim_list[:dim] + [-2, -1] + dim_list[dim + 1 :]
    new_arr = np.transpose(new_arr, dim_list)
    return new_arr


def squeeze_triangular_array(arr, dims=(0, 1)):
    """
    Inverse of `unsqueeze_triangular_array`.

    Parameters
    ----------
    arr : numpy.ndarray
    dims : tuple of int
        The two dimensions to contract to one. These should be contiguous.

    Returns
    -------
    new_arr : numpy.ndarray
        Contracted array
    """
    assert len(dims) == 2
    assert arr.ndim > np.max(dims)
    assert arr.shape[dims[0]] == arr.shape[dims[1]]
    assert dims[1] == dims[0] + 1
    n = arr.shape[dims[0]]
    dim_list = list(range(arr.ndim))
    dim_list = dim_list[: dims[0]] + dim_list[dims[1] + 1 :] + list(dims)
    arr = np.transpose(arr, dim_list)
    new_arr = np.zeros(arr.shape[:-2] + ((n * (n + 1)) // 2,))
    for i in range(n):
        for j in range(i + 1):
            idx = (i * (i + 1)) // 2 + j
            new_arr[..., idx] = arr[..., i, j]
    dim_list = list(range(new_arr.ndim))
    dim_list = dim_list[: dims[0]] + [-1] + dim_list[dims[0] : -1]
    new_arr = np.transpose(new_arr, dim_list)
    return new_arr


def make_high_level_signal_features(
    X,
    fs=1000,
    min_freq=0.0,
    max_freq=55.0,
    directed_spectrum=False,
    csd_params={
        "detrend": "constant",
        "window": "hann",
        "nperseg": 512,
        "noverlap": 256,
        "nfft": None,
    },
):
    """
    Main function: make features from a waveform.

    See ``lpne.unsqueeze_triangular_array`` and ``lpne.squeeze_triangular_array`` to
    convert the power between dense and symmetric forms.

    Parameters
    ----------
    X : numpy.ndarray
        timeseries from which to extract/generate features
        Shape: ``[n_time_steps, n_channel]``
    fs : int, optional
        samplerate
    min_freq : float, optional
        Minimum frequency
    max_freq : float, optional
        Maximum frequency
    directed_spectrum : bool, optional
        Whether to make directed spectrum features
    csd_params : dict, optional
        Parameters sent to ``scipy.signal.csd``

    Returns
    -------
    res : dict
        'power' : numpy.ndarray
            Cross power spectral density features
            Shape: ``[1, n_channel*(n_channel+1)//2, n_freq]``
        'dir_spec' : numpy.ndarray
            Directed spectrum features. Only included if ``directed_spectrum`` is ``True``.
            Shape: ``[1, n_channel, n_channel, n_freq]``
        'freq' : numpy.ndarray
            Frequency bins
            Shape: ``[n_freq]``
    """
    # set up parameters / variables for feature extraction
    n = X.shape[1] # number of time series (i.e. n_channel)
    assert n >= 1, f"{n} < 1"
    X = np.expand_dims(X.T, axis=0)# see https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    assert X.ndim == 3, f"len({X.shape}) != 3"

    # Make cross power spectral density features for each pair of signals.
    nan_mask = np.sum(np.isnan(X), axis=(1, 2)) != 0
    X[nan_mask] = np.random.randn(*X[nan_mask].shape)
    f, cpsd = csd(
        X[:, :, np.newaxis],
        X[:, np.newaxis],
        fs=fs,
        **csd_params,
    )
    i1, i2 = np.searchsorted(f, [min_freq, max_freq])
    f = f[i1:i2]
    cpsd = np.abs(cpsd[..., i1:i2])
    cpsd = squeeze_triangular_array(cpsd, dims=(1, 2))
    cpsd[:, :] *= f  # scale the power features by frequency
    cpsd[nan_mask] = np.nan  # reintroduce NaNs

    # Assemble features.
    res = {
        "power": cpsd,
        "freq": f, 
    }

    # Make directed spectrum features.
    if directed_spectrum:
        f_temp, dir_spec = get_directed_spectrum(X, fs, csd_params=csd_params)
        i1, i2 = np.searchsorted(f, [min_freq, max_freq])
        f_temp = f_temp[i1:i2]
        assert np.allclose(f, f_temp), f"Frequencies don't match:\n{f}\n{f_temp}"
        dir_spec = dir_spec[:, i1:i2] * f_temp.reshape(
            1, -1, 1, 1
        )  # scale by frequency
        dir_spec = np.moveaxis(dir_spec, 1, -1)
        dir_spec[nan_mask] = np.nan  # reintroduce NaNs
        res["dir_spec"] = dir_spec

    return res



# LFP PROCESSING FUNCTIONS/METHODS ##################################################################
"""
Remove artifacts in the LFPs.
References: 
 - https://github.com/carlson-lab/lpne/blob/master/lpne/preprocess/outlier_detection.py
 - https://github.com/carlson-lab/lpne/blob/master/lpne/preprocess/filter.py
 - https://www.johndcook.com/blog/2018/01/02/the-engineers-nyquist-frequency-and-the-sampling-theorem/#:~:text=One%20rule%20of%20thumb%20is,to%20a%20well%2Dknown%20quantity.
 - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

"""
DEFAULT_MAD_TRESHOLD = 15.0
"""Default median absolute deviation threshold for outlier detection"""
LOW_PASS_CUTOFF = 35.0 #hz
"""Default cutoff for low-pass filtering (Hz)"""
LOWCUT = 30.0 # Butterworth bandpass filter parameter
"""Default lowcut for filtering (Hz)"""
HIGHCUT = 55.0 # Butterworth bandpass filter parameter
"""Default highcut for filtering (Hz)"""

Q = 2.0 # Notch filter parameter
"""Notch filter quality parameter"""
ORDER = 3 # Butterworth bandpass filter order
"""Butterworth filter order"""

def _butter_bandpass(lowcut, highcut, fs, order=ORDER):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def _butter_lowpass(cutoff, fs, order=ORDER):
    nyq = 0.5 * fs
    low = cutoff / nyq
    b, a = butter(order, low, btype='lowpass')
    return b, a

def _butter_bandpass_filter(data, lowcut, highcut, fs, order=ORDER):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
def _butter_lowpass_filter(data, cutoff, fs, order=ORDER):
    b, a = _butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def filter_signal_via_bandpass(x, fs, lowcut=LOWCUT, highcut=HIGHCUT, q=Q, order=ORDER,
                                apply_notch_filters=True):
    """
    Apply a bandpass filter and notch filters to the signal.
    Parameters
    ----------
    x : numpy.ndarray
        LFP data
    fs : float
        Samplerate
    lowcut : float, optional
        Lower frequency parameter of bandpass filter
    highcut : float, optional
        Higher frequency parameter of bandpass filter
    q : float, optional
        Notch filter quality factor
    order : int, optional
        Order of bandpass filter
    apply_notch_filter : bool, optional
        Whether to apply the notch filters
    Returns
    -------
    x : numpy.ndarray
    """
    assert x.ndim == 1, f"len({x.shape}) != 1"
    assert lowcut < highcut, f"{lowcut} >= {highcut}"
    # Remove NaNs.
    nan_mask = np.isnan(x)
    x[nan_mask] = 0.0
    # Bandpass.
    x = _butter_bandpass_filter(x, lowcut, highcut, fs, order=order)
    # Remove electrical noise at 60Hz and harmonics.
    if apply_notch_filters:
        for i, freq in enumerate(range(60,int(fs/2),60)):
            b, a = iirnotch(freq, (i+1)*q, fs)
            x = lfilter(b, a, x)
    # Reintroduce NaNs.
    x[nan_mask] = np.nan
    return x
    
def filter_signal_via_lowpass(x, fs, cutoff=LOW_PASS_CUTOFF, q=Q, order=ORDER,
                                apply_notch_filters=True):
    """
    Apply a bandpass filter and notch filters to the signal.
    Parameters
    ----------
    x : numpy.ndarray
        LFP data
    fs : float
        Samplerate
    cutoff : float, optional
        cutoff frequency (Hz) of lowpass filter
    q : float, optional
        Notch filter quality factor
    order : int, optional
        Order of bandpass filter
    apply_notch_filter : bool, optional
        Whether to apply the notch filters
    Returns
    -------
    x : numpy.ndarray
    """
    assert x.ndim == 1, f"len({x.shape}) != 1"
    # Remove NaNs.
    nan_mask = np.isnan(x)
    x[nan_mask] = 0.0
    # Bandpass.
    x = _butter_lowpass_filter(x, cutoff, fs, order=order)
    # Remove electrical noise.
    if apply_notch_filters:
        for i, freq in enumerate(range(60,int(fs/2),60)):
            b, a = iirnotch(freq, (i+1)*q, fs)
            x = lfilter(b, a, x)
    # Reintroduce NaNs.
    x[nan_mask] = np.nan
    return x

def filter_signal(x, fs, cutoff=LOW_PASS_CUTOFF, lowcut=LOWCUT, highcut=HIGHCUT, q=Q, order=ORDER,
                  apply_notch_filters=True, filter_type="bandpass"):
    if filter_type == "bandpass":
        print("general_utils.time_series.filter_signal: NOW APPLYING BANDPASS FILTER")
        return filter_signal_via_bandpass(x, fs, lowcut=lowcut, highcut=highcut, q=q, order=order, apply_notch_filters=apply_notch_filters)
    elif filter_type == "lowpass":
        return filter_signal_via_lowpass(x, fs, cutoff=cutoff, q=q, order=order, apply_notch_filters=apply_notch_filters)
    else:
        raise NotImplementedError()


def mark_outliers(lfps, fs, cutoff=LOW_PASS_CUTOFF, lowcut=LOWCUT, highcut=HIGHCUT,
                    mad_threshold=DEFAULT_MAD_TRESHOLD, filter_type="bandpass"):
    """
    Detect outlying samples in the LFPs.
    Parameters
    ----------
    lfps : dict
        Maps ROI names to LFP waveforms.
    fs : int, optional
        Samplerate
    mad_threshold : float, optional
        A median absolute deviation treshold used to determine whether a point
        is an outlier. A lower value marks more points as outliers.
    Returns
    -------
    lfps : dict
        Maps ROI names to LFP waveforms.
    """
    assert mad_threshold > 0.0, "mad_threshold must be positive!"
    for roi in lfps:
        # Copy the signal.
        trace = np.copy(lfps[roi])
        # Filter the signal.
        trace = filter_signal(
                trace,
                fs,
                cutoff=cutoff, 
                lowcut=lowcut,
                highcut=highcut,
                apply_notch_filters=False,
                filter_type=filter_type, 
        )
        # Subtract out the median and rectify.
        trace = np.abs(trace - np.median(trace))
        # Calculate the MAD and the treshold.
        mad = np.median(trace) # median absolute deviation
        thresh = mad_threshold * mad
        # Mark outlying samples.
        lfps[roi][trace > thresh] = np.nan
    return lfps


def draw_timesteps_to_sample_from(interval_start, interval_stop, window_size, num_samples, nan_locations, max_num_draws=10):
    sample_start_inds = random.sample(range(interval_start, interval_stop - window_size), num_samples)
    replace_attempt_counter = 0
    for i in range(len(sample_start_inds)-1, -1, -1):
        if sample_start_inds[i] in nan_locations or len([loc for loc in nan_locations if sample_start_inds[i] <= loc and sample_start_inds[i]+window_size >= loc]) != 0:
            replace_attempt_counter +=1
            sample_start_inds[i] = None
            for _ in range(max_num_draws):
                new_val = random.sample(range(interval_start, interval_stop - window_size), 1)[0]
                if new_val not in sample_start_inds and new_val not in nan_locations and len([loc for loc in nan_locations if new_val <= loc and new_val+window_size >= loc]) == 0:
                    sample_start_inds[i] = new_val
                    break
            if sample_start_inds[i] is None:
                sample_start_inds.pop(i)
    return sample_start_inds



def draw_timesteps_to_sample_from_using_label_reference(labels, window_size, num_samples, nan_locations, max_num_draws=10):
    sample_start_inds = random.sample(range(len(labels) - window_size), num_samples)
    replace_attempt_counter = 0
    for i in range(len(sample_start_inds)-1, -1, -1):
        if sample_start_inds[i] in nan_locations or len([loc for loc in nan_locations if sample_start_inds[i] <= loc and sample_start_inds[i]+window_size >= loc]) != 0 or sum(labels[sample_start_inds[i]:sample_start_inds[i]+window_size]) != window_size:
            replace_attempt_counter +=1
            sample_start_inds[i] = None
            for _ in range(max_num_draws):
                new_val = random.sample(range(len(labels) - window_size), 1)[0]
                if new_val not in sample_start_inds and new_val not in nan_locations and len([loc for loc in nan_locations if new_val <= loc and new_val+window_size >= loc]) == 0 and sum(labels[new_val:new_val+window_size]) == window_size:
                    sample_start_inds[i] = new_val
                    break
            if sample_start_inds[i] is None:
                sample_start_inds.pop(i)
    return sample_start_inds