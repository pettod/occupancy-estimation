import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

INPUT_FILENAME = "/Users/todorov/Documents/ef/occupancy-ml/data/0_input/20241104_105000_cafe/20241104_111000.WAV"  # Change this to the path of your WAV file
OUTPUT_FILENAME = "20241104_111000.WAV"
CUTOFF_FREQ = 10000  # Adjust this to your desired cutoff frequency
PLOT = False  # Super slow


def highpassFilter(data, cutoff_freq, sample_rate, order=4):
    # Normalize the frequency
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    
    # Create a Butterworth high-pass filter
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    
    # Apply the filter to the data using filtfilt for zero-phase filtering
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data


def plotWaveformAndFrequency(time, original_data, filtered_data, sample_rate, title="Audio Waveforms"):
    # Time domain plot
    plt.figure(figsize=(12, 10))
    
    # Plot original and filtered signals in time domain
    plt.subplot(4, 1, 1)
    plt.plot(time, original_data, label="Original Signal")
    plt.title(f"Original time domain")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(time, filtered_data, label="Filtered Signal", color="orange")
    plt.title(f"Filtered time domain")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()

    # Frequency domain plot
    N = len(original_data)
    freqs = np.fft.fftfreq(N, 1 / sample_rate)
    original_fft = np.fft.fft(original_data)
    filtered_fft = np.fft.fft(filtered_data)

    # Plot only the positive frequencies (real part)
    plt.subplot(2, 2, 3)
    plt.plot(freqs[:N//2], np.abs(original_fft)[:N//2], label="Original Signal (FFT)")
    plt.title(f"Original frequency domain")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()

    # Plot only the positive frequencies (real part)
    plt.subplot(2, 2, 4)
    plt.plot(freqs[:N//2], np.abs(filtered_fft)[:N//2], label="Filtered Signal (FFT)", color="orange")
    plt.title(f"Filtered frequency domain")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def main():
    # Load the input WAV file
    data, sample_rate = sf.read(INPUT_FILENAME)

    # Mono
    if data.ndim == 1:  # Mono
        filtered_data = highpassFilter(data, cutoff_freq=CUTOFF_FREQ, sample_rate=sample_rate)  # Example cutoff at 100 Hz
    # Stereo
    else:
        filtered_data = np.apply_along_axis(highpassFilter, 0, data, cutoff_freq=CUTOFF_FREQ, sample_rate=sample_rate)

    # Create a time axis for plotting (based on the sample rate and number of samples)
    time = np.arange(data.shape[0]) / sample_rate

    if PLOT:
        plotWaveformAndFrequency(time, data, filtered_data, sample_rate, title="High-pass Filter Effect")
    sf.write(OUTPUT_FILENAME, filtered_data, sample_rate)
    print(f"Filtered audio saved as {OUTPUT_FILENAME}")


main()
