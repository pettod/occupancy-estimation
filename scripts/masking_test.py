import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import augment


# Load a sample audio file
sample_rate = 192000
duration = 60
num_frames = int(duration * sample_rate)
frame_offset = 0
n_mels = 64
freq_mask_param = 10
time_mask_param = 6000
audio_waveform, sample_rate = torchaudio.load(
    "/Users/todorov/Documents/ef/occupancy-estimation/data_high-pass/Audiomoth_10430017/20241119/Office_0_64_Tuesday/20241119_101552.WAV",
    frame_offset=frame_offset,
    num_frames=num_frames,
)
new_sample_rate = sample_rate//3
resampler = T.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
audio_waveform = resampler(audio_waveform)
sample_rate = new_sample_rate

# Apply MelSpectrogram transformation
mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
spectrogram = mel_transform(audio_waveform)

# Apply TimeMasking
#time_masking = T.TimeMasking(time_mask_param=time_mask_param)  # Mask up to 30 time steps
#masked_spectrogram = time_masking(spectrogram)

# Apply FrequencyMasking
#frequency_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)  # Mask up to 10 frequency bins
#masked_spectrogram = frequency_masking(masked_spectrogram)

masked_spectrogram = augment(spectrogram)

# Plot original and masked spectrograms
plt.figure(figsize=(12, 6))

# Original Spectrogram
plt.subplot(1, 2, 1)
plt.title("Original Spectrogram")
plt.imshow(spectrogram[0].log2().detach().numpy(), aspect='auto', origin='lower')
plt.colorbar(label="Log-Scaled Amplitude")

# Calculate time and frequency values
time_points = np.linspace(0, duration, spectrogram.shape[2])
freq_points = np.linspace(0, sample_rate//2, spectrogram.shape[1])

plt.xticks(np.linspace(0, spectrogram.shape[2], 5), [f"{t:.1f}" for t in np.linspace(0, duration, 5)])
plt.yticks(np.linspace(0, spectrogram.shape[1], 6), [f"{f/1000:.1f}" for f in np.linspace(0, sample_rate//2, 6)])
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (kHz)")

# Masked Spectrogram  
plt.subplot(1, 2, 2)
plt.title("Masked Spectrogram")
plt.imshow(masked_spectrogram[0].log2().detach().numpy(), aspect='auto', origin='lower')
plt.colorbar(label="Log-Scaled Amplitude")
plt.xticks(np.linspace(0, masked_spectrogram.shape[2], 5), [f"{t:.1f}" for t in np.linspace(0, duration, 5)])
plt.yticks(np.linspace(0, masked_spectrogram.shape[1], 6), [f"{f/1000:.1f}" for f in np.linspace(0, sample_rate//2, 6)])
plt.xlabel("Time (seconds)") 
plt.ylabel("Frequency (kHz)")

plt.tight_layout()
plt.show()
