import random
from glob import glob

import numpy as np
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T


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
            self, input_paths, target_paths=None, transform=None,
            input_normalize=None, sample_rate=192000, mel_bands=256):
        self.input_audio_paths = readAudioPaths(input_paths)
        self.target_occupancy_paths = readCsvPaths(target_paths)
        self.transform = transform or T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=mel_bands)
        self.input_normalize = input_normalize
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.input_audio_paths)

    def __getitem__(self, i):
        seed = random.randint(0, 2**32)
        audio_waveform, original_sample_rate = torchaudio.load(self.input_audio_paths[i])
        
        # Resample the audio if necessary
        if original_sample_rate != self.sample_rate:
            resampler = T.Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)
            audio_waveform = resampler(audio_waveform)
        
        # Transform the waveform to a spectrogram
        spectrogram = self.transform(audio_waveform)
        
        return spectrogram


if __name__ == "__main__":
    input_paths = [
        "data/0_input/20241016_155159.WAV",
        "data/0_input/20241012_061010.WAV"
    ]
    dataset = AudioSpectrogramDataset(input_paths)
    spectrogram = dataset[1]
    print(spectrogram.shape)  # Output: (number_of_mels, time_steps)
    # https://en.wikipedia.org/wiki/Mel_scale
