import random
from glob import glob
import numpy as np
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import json
import torch
import matplotlib.pyplot as plt


def readAudio(audio_path, transform, seed):
    samplerate, audio = wavfile.read(audio_path)
    # TODO: Make spetral image
    return audio


def readAudioPaths(data_path):
    if data_path is None or type(data_path) == list:
        return data_path
    return np.array(sorted(glob(f"{data_path}/*.WAV")))


def readCsvPaths(data_path):
    if data_path is None or type(data_path) == list:
        return data_path
    return np.array(sorted(glob(f"{data_path}/*.csv")))


def readJson(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


class ImageDataset(Dataset):
    def __init__(
            self, input_path, target_path=None, transform=None,
            input_normalize=None):
        self.input_audio_paths = readAudioPaths(input_path)
        self.target_occupancy_paths = readCsvPaths(target_path)
        self.transform = transform
        self.input_normalize = input_normalize

    def __len__(self):
        return len(self.input_audio_paths)

    def __getitem__(self, i):
        seed = random.randint(0, 2**32)
        input_image = readAudio(self.input_audio_paths[i], self.transform, seed)
        if self.input_normalize:
            input_image = self.input_normalize(input_image)
        if self.target_occupancy_paths is not None:
            target_image = readAudio(self.target_occupancy_paths[i], self.transform, seed)
            return input_image, target_image
        return input_image, self.input_audio_paths[i]


class AudioSpectrogramDataset(Dataset):
    def __init__(
            self, dataset_path, transform=None,
            input_normalize=None, sample_rate=192000, mel_bands=256,
        ):
        self.dataset = readJson(dataset_path)
        self.transform = transform or T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=mel_bands)
        self.input_normalize = input_normalize
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        seed = random.randint(0, 2**32)
        audio_file_path = self.dataset[i]["audio_file_path"]
        occupancy = torch.tensor(int(self.dataset[i]["occupancy"]), dtype=torch.float32)
        start_time = self.dataset[i]["start_time"]
        duration = self.dataset[i]["end_time"] - start_time
        num_frames = int(duration * self.sample_rate)
        frame_offset = int(start_time * self.sample_rate)
        audio_waveform, original_sample_rate = torchaudio.load(
            audio_file_path,
            frame_offset=frame_offset,
            num_frames=num_frames,
        )
        
        # Resample the audio if necessary
        if original_sample_rate != self.sample_rate:
            raise ValueError(
                f"Sampling rate mismatch: expected {self.sample_rate} Hz, but" +
                " got {original_sample_rate} Hz"
            )
            #resampler = T.Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)
            #audio_waveform = resampler(audio_waveform)
        
        # Transform the waveform to a spectrogram
        spectrogram = self.transform(audio_waveform)
        
        return spectrogram, occupancy


if __name__ == "__main__":
    dataset_file_path = "dataset_2-0s.json"
    dataset = AudioSpectrogramDataset(dataset_file_path)
    # https://en.wikipedia.org/wiki/Mel_scale
    spectrogram, occupancy = dataset[0]
    print(spectrogram.shape)  # (number_of_mels, time_steps)
    print(occupancy, "people")

    plt.imshow(spectrogram.log2()[0, :, :].numpy(), cmap="viridis")
    plt.show()
