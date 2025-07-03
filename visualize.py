import librosa
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable

def save_spectrogram(waveform: np.ndarray, sr: int | float, labels: Iterable[str], name: str, output_dir: str):
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)

    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log', cmap='winter')
    plt.title(f"{name} - Labels: [{', '.join(labels)}]", wrap=True)
    plt.colorbar(label='dB')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}.png")
