import librosa
import numpy as np

__all__ = [
    'centroid_variance',
    'bndwidth_mean',
    'zcr_mean',
    'rms_variance',
    'rolloff_median'
]

def centroid_variance(waveform, sr):
    """
    Calculate the spectral centroid variance of a waveform.
    
    Parameters:
    waveform (np.ndarray): The audio signal.
    sr (int): The sample rate of the audio signal.
    
    Returns:
    float: The variance of the spectral centroids.
    """
    spectral_centroids = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
    return np.var(spectral_centroids)

def bandwidth_mean(waveform, sr):
    """
    Calculate the mean spectral bandwidth of a waveform.
    
    Parameters:
    waveform (np.ndarray): The audio signal.
    sr (int): The sample rate of the audio signal.
    
    Returns:
    float: Mean spectral bandwidth.
    """
    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
    return np.mean(bandwidth)

def zcr_mean(waveform, sr):
    """
    Calculate the mean zero crossing rate of a waveform.
    
    Parameters:
    waveform (np.ndarray): The audio signal.
    sr (int): The sample rate.
    
    Returns:
    float: Mean zero crossing rate.
    """
    zcr = librosa.feature.zero_crossing_rate(y=waveform)[0]
    return np.mean(zcr)


def rms_variance(waveform, sr):
    """
    Calculate the variance of RMS energy over the waveform.
    
    Parameters:
    waveform (np.ndarray): The audio signal.
    sr (int): The sample rate.
    
    Returns:
    float: Variance of RMS energy.
    """
    rms = librosa.feature.rms(y=waveform)[0]
    return np.var(rms)


def rolloff_median(waveform, sr):
    """
    Calculate the median spectral roll-off frequency of a waveform.
    
    Parameters:
    waveform (np.ndarray): The audio signal.
    sr (int): The sample rate.
    
    Returns:
    float: Median roll-off frequency.
    """
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
    return np.median(rolloff)