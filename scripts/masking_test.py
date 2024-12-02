import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

# Load a sample audio file
waveform, sample_rate = torchaudio.load("20241016_155159_original.WAV")

# Apply MelSpectrogram transformation
mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=16)
spectrogram = mel_transform(waveform)

# Apply TimeMasking
time_masking = T.TimeMasking(time_mask_param=300)  # Mask up to 30 time steps
masked_spectrogram = time_masking(spectrogram)

# Apply FrequencyMasking
frequency_masking = T.FrequencyMasking(freq_mask_param=2)  # Mask up to 10 frequency bins
masked_spectrogram = frequency_masking(masked_spectrogram)

# Plot original and masked spectrograms
plt.figure(figsize=(12, 6))

# Original Spectrogram
plt.subplot(1, 2, 1)
plt.title("Original Spectrogram")
plt.imshow(spectrogram[0].log2().detach().numpy(), aspect='auto', origin='lower')
plt.colorbar(label="Log-Scaled Amplitude")
plt.xlabel("Time")
plt.ylabel("Frequency")

# Masked Spectrogram
plt.subplot(1, 2, 2)
plt.title("Masked Spectrogram")
plt.imshow(masked_spectrogram[0].log2().detach().numpy(), aspect='auto', origin='lower')
plt.colorbar(label="Log-Scaled Amplitude")
plt.xlabel("Time")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
