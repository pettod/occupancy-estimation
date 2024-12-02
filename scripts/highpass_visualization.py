import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, spectrogram

def highpass_filter(data, sample_rate, cutoff=1000, order=5):
    """Applies a high-pass filter to the audio data."""
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# Load the WAV file
file_path = "20241016_155159_original.WAV"  # Replace with your file path
sample_rate, data = wavfile.read(file_path)

# Normalize if stereo
if len(data.shape) == 2:  # Stereo
    data = data.mean(axis=1)  # Convert to mono by averaging channels

# Apply high-pass filter
cutoff_frequency = 10000  # Hz
filtered_data = highpass_filter(data, sample_rate, cutoff=cutoff_frequency)

# Compute spectrograms
def compute_spectrogram(data, sample_rate):
    f, t, Sxx = spectrogram(data, fs=sample_rate, nperseg=1024, noverlap=512)
    return f, t, Sxx

frequencies, times, original_spec = compute_spectrogram(data, sample_rate)
_, _, filtered_spec = compute_spectrogram(filtered_data, sample_rate)

# Compute residual spectrogram
residual_spec = np.abs(original_spec - filtered_spec)

# Find global intensity range across all spectrograms
min_val = min(original_spec.min(), filtered_spec.min(), residual_spec.min())
max_val = max(original_spec.max(), filtered_spec.max(), residual_spec.max())

# Plot spectrograms with consistent scale
def plot_spectrogram(ax, frequencies, times, spectrogram, title, vmin, vmax):
    img = ax.pcolormesh(times, frequencies, 10 * np.log10(spectrogram + 1e-10), 
                        shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.colorbar(img, ax=ax, label="Intensity (dB)")

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
plot_spectrogram(axes[0], frequencies, times, original_spec, 
                 "Original Audio Spectrogram", 10 * np.log10(min_val + 1e-10), 10 * np.log10(max_val + 1e-10))
plot_spectrogram(axes[1], frequencies, times, filtered_spec, 
                 "High-Pass Filtered Spectrogram", 10 * np.log10(min_val + 1e-10), 10 * np.log10(max_val + 1e-10))
plot_spectrogram(axes[2], frequencies, times, residual_spec, 
                 "Residual Spectrogram (Original - Filtered)", 10 * np.log10(min_val + 1e-10), 10 * np.log10(max_val + 1e-10))

# Display the plots
plt.tight_layout()
plt.show()
