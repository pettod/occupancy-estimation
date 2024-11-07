import random
from glob import glob

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset


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
