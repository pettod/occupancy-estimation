import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from collections import deque
import random
from importlib import import_module
import torch
import os

from src.utils.utils import loadModel
from scripts.highpass_filter import highpassFilter


# Audio stream parameters
SAMPLING_RATE = 44100
AUDIO_BUFFER = 1024             # Frames per buffer
AUDIO_FORMAT = pyaudio.paInt16  # 16-bit integer
MODEL_PATH = "saved_models/2024-11-20_110334_60s-window_1500-samples"

AUDIO_CHANNELS = 1              # Mono
CUTOFF = 10000
INTERVAL = 30
COUNT_HISTORY_SAMPLES = 100
COUNT_SMOOTHENED_SAMPLES = 10  # Min 1
SMOOTH_FFT = False


class Model():
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        with torch.no_grad():
            self.model, config = loadModelAndConfig(model_path, self.device)
        self.sample_rate = config.SAMPLE_RATE
        self.transform = config.TRANSFORM
        self.input_normalize = config.INPUT_NORMALIZE

    def __call__(self, original_audio):
        with torch.no_grad():
            filtered_audio = highpassFilter(original_audio, CUTOFF, self.sample_rate)
            x = torch.from_numpy(filtered_audio.copy()).float().to(self.device).unsqueeze(0)
            
            ##### Remove this random tensor #####
            x = torch.rand(1, 11520000) * 2 - 1
            ##### ######################### #####
            
            x = self.transform(x)
            x = self.input_normalize(x)
            pred = self.model(x.unsqueeze(0)).squeeze(0).cpu().numpy()
            pred = np.argmax(pred).astype(int)
        return pred


def loadModelAndConfig(model_path, device):
    config = import_module(os.path.join(
        model_path, "codes.config").replace("/", ".")).CONFIG
    model = config.MODELS[0].to(device)
    loadModel(model, model_path=model_path)
    return model, config


def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode="same")


class MicrophonePlot:
    def __init__(self, model=None):
        # Open audio stream from the microphone
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=SAMPLING_RATE,
            input=True,
            frames_per_buffer=AUDIO_BUFFER,
        )
        self.model = model
        self.is_paused = False
        self.count_history_samples = deque(maxlen=COUNT_SMOOTHENED_SAMPLES)
        self.count_history_smoothened_samples = deque(maxlen=COUNT_HISTORY_SAMPLES)
        for i in range(COUNT_SMOOTHENED_SAMPLES): self.count_history_samples.append(0)
        for i in range(COUNT_HISTORY_SAMPLES): self.count_history_smoothened_samples.append(0)
        self.setFigures()

    def predictCount(self, audio_clip):
        if self.model:
            prediction = self.model(audio_clip)
        else:
            prediction = random.randint(0, 50)
        return prediction

    def setFigures(self):
        # Figure and axes
        self.fig = plt.figure(figsize=(8, 6))
        self.gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        self.ax1 = self.fig.add_subplot(self.gs[0, 0])
        self.ax2 = self.fig.add_subplot(self.gs[1, 0])
        self.ax3 = self.fig.add_subplot(self.gs[0, 1]) #self.gs[:, 1])
        self.ax4 = self.fig.add_subplot(self.gs[1, 1])

        # ax1 plot
        self.ax1.set_ylim(-3000, 3000)
        self.ax1.set_xlim(0, AUDIO_BUFFER)
        self.ax1.set_title("Time Domain")
        self.ax1.set_xlabel("Samples")
        self.ax1.set_ylabel("Amplitude")
        self.ax2.set_yscale("log")
        self.plot_time_data, = self.ax1.plot(np.arange(0, AUDIO_BUFFER), np.zeros(AUDIO_BUFFER))

        # ax2 plot
        self.ax2.set_xlim(20, SAMPLING_RATE//2)  # Nyquist frequency (SAMPLING_RATE/2)
        self.ax2.set_ylim(0, 10**6)
        self.ax2.set_title("Frequency Domain")
        self.ax2.set_xlabel("Frequency (Hz)")
        self.ax2.set_ylabel("Magnitude")
        self.x_freq = np.fft.fftfreq(AUDIO_BUFFER, 1/SAMPLING_RATE)[:AUDIO_BUFFER//2]  # Only positive frequencies
        self.plot_frequency_data, = self.ax2.plot(self.x_freq, np.zeros(AUDIO_BUFFER//2))

        # ax3 plot
        self.ax3.set_title("Count")
        self.ax3.axis("off")
        self.plot_count_count = self.ax3.text(0.5, 0.5, "", fontsize=170, ha="center", va="center")

        # ax4 plot
        self.ax4.set_title("Count History")
        self.ax4.set_xlabel("Sample")
        self.ax4.set_ylabel("Count")
        self.count_history, = self.ax4.plot(self.count_history_smoothened_samples)
        self.ax4.set_ylim(0, 51)

    def update_plots(self, frame):
        # Read AUDIO_BUFFER-size data
        data = np.frombuffer(self.stream.read(AUDIO_BUFFER, exception_on_overflow=False), dtype=np.int16)
        self.plot_time_data.set_ydata(data)
        
        # FFT
        fft_data = np.abs(np.fft.fft(data))[:AUDIO_BUFFER//2]
        fft_data += 1
        if SMOOTH_FFT:
            fft_data = moving_average(fft_data, window_size=9)
        self.plot_frequency_data.set_ydata(fft_data)
        
        # Count
        count_count = self.predictCount(data)
        self.plot_count_count.set_text(count_count)
        self.count_history_samples.append(count_count)
        self.count_history_smoothened_samples.append(round(np.mean(self.count_history_samples)))
        self.count_history.set_ydata(self.count_history_smoothened_samples)
        
        return self.plot_time_data, self.plot_frequency_data, self.plot_count_count, self.count_history

    def pauseAnimation(self, event):
        if event.key == " ":  # Check if the space bar was pressed
            if self.is_paused:
                self.animation.event_source.start()  # Resume the animation
            else:
                self.animation.event_source.stop()   # Pause the animation
            self.is_paused = not self.is_paused

    def run(self):
        # Pause/Resume playing
        self.fig.canvas.mpl_connect("key_press_event", self.pauseAnimation)

        # Update plot
        self.animation = FuncAnimation(self.fig, self.update_plots, blit=True, interval=INTERVAL)

        # Plot
        plt.tight_layout()
        plt.show()

        # Close the audio stream
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


def main():
    model = Model(MODEL_PATH)
    m = MicrophonePlot(model)
    m.run()


main()
