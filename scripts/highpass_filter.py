import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from glob import glob
from datetime import datetime
import os
from tqdm import tqdm


"""Expected data structure
DATA_DIR
    Recording device name/barcode
        Date
            Recording description
                WAV files
"""

INPUT_DATA_DIR = "data_original"
CUTOFF_FREQ = 10000
ORDER = 15
OUTPUT_DATA_DIR = f"data_high-pass_{CUTOFF_FREQ}Hz-cutoff_{ORDER}th-order"
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
    input_dirs = glob(os.path.join(INPUT_DATA_DIR, "*/*/*/"))
    existing_output_dirs = set(glob(os.path.join(OUTPUT_DATA_DIR, "*/*/*")))
    filtered_input_dirs = [d for d in input_dirs if os.path.join(OUTPUT_DATA_DIR, os.path.relpath(d, INPUT_DATA_DIR)) not in existing_output_dirs]
    print(f"Found {len(filtered_input_dirs)} directories to process:")
    for i, f in enumerate(filtered_input_dirs, start=1):
        print(f"{i}. {f}")
    confirm_to_continue = input("Do you want to continue (y/n): ")
    if confirm_to_continue.lower() != "y":
        return

    for recording_dir in tqdm(filtered_input_dirs):
        os.makedirs(os.path.join(OUTPUT_DATA_DIR, *recording_dir.split(os.sep)[1:]), exist_ok=True)
        wav_paths = glob(os.path.join(recording_dir, "*.WAV"))
        for wav_path in tqdm(sorted(wav_paths)):
            data, sample_rate = sf.read(wav_path)
            output_filename = os.path.join(OUTPUT_DATA_DIR, *wav_path.split(os.sep)[1:])

            # Mono
            if data.ndim == 1:  # Mono
                filtered_data = highpassFilter(data, cutoff_freq=CUTOFF_FREQ, sample_rate=sample_rate, order=ORDER)  # Example cutoff at 100 Hz
            # Stereo
            else:
                filtered_data = np.apply_along_axis(highpassFilter, 0, data, cutoff_freq=CUTOFF_FREQ, sample_rate=sample_rate, order=ORDER)

            # Create a time axis for plotting (based on the sample rate and number of samples)
            time = np.arange(data.shape[0]) / sample_rate

            if PLOT:
                plotWaveformAndFrequency(time, data, filtered_data, sample_rate, title="High-pass Filter Effect")
            try:
                sf.write(output_filename, filtered_data, sample_rate)
            except sf.LibsndfileError:
                print(f"Error writing file {output_filename}")
                print("Disk may be full")
                continue
    print(f"Filtered audio files saved to {OUTPUT_DATA_DIR}")


if __name__ == "__main__":
    main()
