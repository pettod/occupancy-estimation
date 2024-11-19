from src.dataset import AudioSpectrogramDataset as Dataset
import torch
from tqdm import tqdm
from config import CONFIG


TRAIN_FILE = "dataset_60-0s_train.json"
REPLACED_DATA_PATH_ROOT = "data_high-pass"
TRAIN_DATASET = Dataset(
    TRAIN_FILE,
    REPLACED_DATA_PATH_ROOT,
    transform=CONFIG.TRANSFORM,
)


def main():
    samples = []
    for spectrogram, occupancy in tqdm(TRAIN_DATASET):
        samples.append(spectrogram)
    samples = torch.stack(samples)
    std = torch.std(samples)
    mean = torch.mean(samples)
    print("Mean:", mean)
    print("Std:", std)


main()
