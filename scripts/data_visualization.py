import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import cpu_count
import torchaudio.transforms as T

# Project files
from src.dataset import AudioSpectrogramDataset as Dataset


# Data paths
DATA_FILENAME = "dataset_60-0s_valid.json"
REPLACED_DATA_PATH_ROOT = "data_high-pass"

# Model parameters
BATCH_SIZE = 16
DEVICE = torch.device("mps")
NUM_WORKERS = 0  #cpu_count()


def predictResults():
    # Dataset
    sample_rate = 192000
    transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=64, n_fft=8000)
    input_normalize = None
    dataset = Dataset(DATA_FILENAME, REPLACED_DATA_PATH_ROOT, transform, input_normalize, sample_rate, True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Predict and save
    for i, (x, y, file_info) in enumerate(tqdm(dataloader)):
            #plt.imshow(x[0].squeeze(0).numpy(), aspect='auto', origin='lower')
            #plt.specgram(x[0].squeeze(0).numpy(), NFFT=1024, Fs=sample_rate, noverlap=512, cmap='viridis')
            #plt.plot(x[0].squeeze(0).numpy())
            #plt.title(y[0])
            #plt.show()
            x, y = x.to(DEVICE), y.numpy()
            for i, (gt, x_) in enumerate(zip(y, x)):
                gt = gt.astype(int)


if __name__ == "__main__":
    predictResults()
