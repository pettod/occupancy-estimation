import random
from torch.utils.data import Dataset
import torchaudio
import json
import torch
import matplotlib.pyplot as plt
import os
import torchaudio.transforms as T


FILTERED_DATA_PATH = None  #"data_original/Audiomoth_10488200"


def readJson(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    if FILTERED_DATA_PATH:
        data = [d for d in data if FILTERED_DATA_PATH in d["audio_file_path"]]
    return data


def replaceRootPath(original_path, new_root):
    path_parts = original_path.split(os.sep)
    transformed_path = os.path.join(new_root, *path_parts[1:])
    return transformed_path


def augment(spectrogram, probability=0.7, time_mask_param=1000, freq_mask_param=3, number_of_masks=6):
    # Time domain augmentations
    if random.random() < probability:
        for _ in range(number_of_masks):
            time_mask = T.TimeMasking(time_mask_param=time_mask_param)
            spectrogram = time_mask(spectrogram)

    # Frequency domain augmentations
    if random.random() < probability:
        for _ in range(number_of_masks):
            freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
            spectrogram = freq_mask(spectrogram)
        
    return spectrogram


class AudioSpectrogramDataset(Dataset):
    def __init__(
            self, dataset_path, replaced_data_path_root=None, transform=None,
            input_normalize=None, sample_rate=192000, get_file_info=False,
            augment=True,
        ):
        self.dataset = readJson(dataset_path)
        self.replaced_data_path_root = replaced_data_path_root
        self.transform = transform
        self.input_normalize = input_normalize
        self.sample_rate = sample_rate
        self.get_file_info = get_file_info
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        seed = random.randint(0, 2**32)
        audio_file_path = self.dataset[i]["audio_file_path"]
        start_time = self.dataset[i]["start_time"]
        end_time = self.dataset[i]["end_time"]
        occupancy = torch.tensor(int(self.dataset[i]["occupancy"]), dtype=torch.float32)
        file_info = {
            "audio_file_path": audio_file_path,
            "start_time": start_time,
            "end_time": end_time,
        }
        if self.replaced_data_path_root:
            audio_file_path = replaceRootPath(audio_file_path, self.replaced_data_path_root)
        file_info["replaced_audio_file_path"] = audio_file_path
        duration = end_time - start_time
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
        
        # Resample, take only third of the samples
        resampler = T.Resample(orig_freq=original_sample_rate, new_freq=original_sample_rate//3)
        audio_waveform = resampler(audio_waveform)

        # Transform the waveform to a spectrogram
        spectrogram = self.transform(audio_waveform)
        if self.input_normalize:
            spectrogram = self.input_normalize(spectrogram)
        #spectrogram[:] = occupancy
        if self.augment:
            spectrogram = augment(spectrogram)
        if self.get_file_info:
            return spectrogram, occupancy, file_info
        else:
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
