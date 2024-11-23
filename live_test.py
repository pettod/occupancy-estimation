import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from collections import deque
import random


# Audio stream parameters
AUDIO_FORMAT = pyaudio.paInt16  # 16-bit integer
AUDIO_CHANNELS = 1              # Mono
SAMPLING_RATE = 44100
AUDIO_BUFFER = 1024             # Frames per buffer
SMOOTH_FFT = False
OCCUPANCY_HISTORY_SAMPLES = 100
INTERVAL = 30


def predictOccupancy(audio_clip):
    return random.randint(0, 50)

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode="same")


class MicrophonePlot:
    def __init__(self):
        # Open audio stream from the microphone
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=SAMPLING_RATE,
            input=True,
            frames_per_buffer=AUDIO_BUFFER,
        )
        self.is_paused = False
        self.occupancy_history_samples = deque(maxlen=OCCUPANCY_HISTORY_SAMPLES)
        for i in range(OCCUPANCY_HISTORY_SAMPLES): self.occupancy_history_samples.append(0)
        self.setFigures()

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
        self.ax3.set_title("Occupancy")
        self.ax3.axis("off")
        self.plot_occupancy_count = self.ax3.text(0.5, 0.5, "", fontsize=170, ha="center", va="center")

        # ax4 plot
        self.ax4.set_title("Occupancy History")
        self.ax4.set_xlabel("Sample")
        self.ax4.set_ylabel("Occupancy")
        self.occupancy_history, = self.ax4.plot(self.occupancy_history_samples)
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
        
        # Occupancy
        occupancy_count = predictOccupancy(data)
        self.plot_occupancy_count.set_text(occupancy_count)
        self.occupancy_history_samples.append(occupancy_count)
        self.occupancy_history.set_ydata(self.occupancy_history_samples)
        
        return self.plot_time_data, self.plot_frequency_data, self.plot_occupancy_count, self.occupancy_history

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
    m = MicrophonePlot()
    m.run()


main()
