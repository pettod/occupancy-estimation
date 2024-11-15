from src.dataset import AudioSpectrogramDataset as Dataset
import torch
from tqdm import tqdm


TRAIN_FILE = "dataset_2-0s_train.json"
REPLACED_DATA_PATH_ROOT = "data_high-pass"
TRAIN_DATASET = Dataset(TRAIN_FILE, REPLACED_DATA_PATH_ROOT)


def main():
    mean = 0
    std = []
    for spectrogram, occupancy in tqdm(TRAIN_DATASET):
        mean += torch.mean(spectrogram)
        std.append(spectrogram)
    std = torch.std(torch.stack(std))
    print("Mean:", mean)
    print("Std:", std)


main()
