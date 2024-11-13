import random
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import json
import torch
import matplotlib.pyplot as plt
import os


def readJson(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def replaceRootPath(original_path, new_root):
    path_parts = original_path.split(os.sep)
    transformed_path = os.path.join(new_root, *path_parts[1:])
    return transformed_path


class AudioSpectrogramDataset(Dataset):
    def __init__(
            self, dataset_path, replaced_data_path_root=None, transform=None,
            input_normalize=None, sample_rate=192000, mel_bands=256,
        ):
        self.dataset = readJson(dataset_path)
        self.replaced_data_path_root = replaced_data_path_root
        self.transform = transform or T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=mel_bands)
        self.input_normalize = input_normalize
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        seed = random.randint(0, 2**32)
        audio_file_path = self.dataset[i]["audio_file_path"]
        if self.replaced_data_path_root:
            audio_file_path = replaceRootPath(audio_file_path, self.replaced_data_path_root)
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
