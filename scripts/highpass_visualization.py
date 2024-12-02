import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, freqz, spectrogram

def highpass_filter(data, sample_rate, cutoff=1000, order=4):
    """Applies a high-pass filter to the audio data."""
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a, filtfilt(b, a, data)

# Load the WAV file
file_path = "20241016_155159_original.WAV"  # Replace with your file path
sample_rate, data = wavfile.read(file_path)

# Normalize if stereo
if len(data.shape) == 2:  # Stereo
    data = data.mean(axis=1)  # Convert to mono by averaging channels

# Apply high-pass filter
cutoff_frequency = 10000  # Hz
b, a, filtered_data = highpass_filter(data, sample_rate, cutoff=cutoff_frequency)

# Compute filter frequency response
w, h = freqz(b, a, worN=8000, fs=sample_rate)

# Compute spectrograms
def compute_spectrogram(data, sample_rate):
    f, t, Sxx = spectrogram(data, fs=sample_rate, nperseg=1024, noverlap=512)
    return f, t, Sxx

frequencies, times, original_spec = compute_spectrogram(data, sample_rate)
_, _, filtered_spec = compute_spectrogram(filtered_data, sample_rate)
residual_spec = np.abs(original_spec - filtered_spec)

# Set up the grid layout
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 1, width_ratios=[1], height_ratios=[1, 2])

# Filter frequency response
ax1 = fig.add_subplot(gs[0])
ax1.plot(w, 20 * np.log10(abs(h)), 'b')
ax1.set_title("High-Pass Filter Frequency Response")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Gain (dB)")
ax1.set_xlim(0, cutoff_frequency * 2)
ax1.set_ylim(-210, 10)
ax1.grid()

# Spectrogram subplots
spectrogram_axes = gs[1].subgridspec(1, 3)

# Plot spectrograms
def plot_spectrogram(ax, frequencies, times, spectrogram, title, vmin, vmax):
    img = ax.pcolormesh(times, frequencies, 10 * np.log10(spectrogram + 1e-10), 
                        shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.colorbar(img, ax=ax, label="Intensity (dB)")

# Find global intensity range
min_val = min(original_spec.min(), filtered_spec.min(), residual_spec.min())
max_val = max(original_spec.max(), filtered_spec.max(), residual_spec.max())

ax2 = fig.add_subplot(spectrogram_axes[0])
plot_spectrogram(ax2, frequencies, times, original_spec, 
                 "Original Audio Spectrogram", 10 * np.log10(min_val + 1e-10), 10 * np.log10(max_val + 1e-10))

ax3 = fig.add_subplot(spectrogram_axes[1])
plot_spectrogram(ax3, frequencies, times, filtered_spec, 
                 "High-Pass Filtered Spectrogram", 10 * np.log10(min_val + 1e-10), 10 * np.log10(max_val + 1e-10))

ax4 = fig.add_subplot(spectrogram_axes[2])
plot_spectrogram(ax4, frequencies, times, residual_spec, 
                 "Residual Spectrogram (Original - Filtered)", 10 * np.log10(min_val + 1e-10), 10 * np.log10(max_val + 1e-10))

# Adjust layout
plt.tight_layout()
plt.show()
