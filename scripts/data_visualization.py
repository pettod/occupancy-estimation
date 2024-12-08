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
from config import CONFIG


# Data paths
DATA_FILENAME = "dataset_60-0s_valid.json"
REPLACED_DATA_PATH_ROOT = "data_high-pass"

# Model parameters
BATCH_SIZE = 16
DEVICE = torch.device("mps")
NUM_WORKERS = cpu_count()


def predictResults():
    # Dataset
    sample_rate = CONFIG.SAMPLE_RATE
    transform = CONFIG.TRANSFORM
    input_normalize = None
    dataset = Dataset(DATA_FILENAME, REPLACED_DATA_PATH_ROOT, transform, input_normalize, sample_rate, True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    gt_means = {}
    gt_stds = {}
    gt_means_divideded_by_stds = {}

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
            mean = x_.mean().cpu().numpy()
            std = x_.std().cpu().numpy()
            gmdbs = mean / std
            if gt in gt_means:
                gt_means[gt].append(mean)
                gt_stds[gt].append(std)
                gt_means_divideded_by_stds[gt].append(gmdbs)
            else:
                gt_means[gt] = [mean]
                gt_stds[gt] = [std]
                gt_means_divideded_by_stds[gt] = [gmdbs]

    return gt_means, gt_stds, gt_means_divideded_by_stds


if __name__ == "__main__":
    means, stds, gmdbs = predictResults()

    alpha = 0.15
    plt.subplot(3, 1, 1)
    x = []
    y = []
    for key, values in means.items():
        x.extend([key] * len(values))
        y.extend(values)
    plt.scatter(x, y, color="blue", alpha=alpha)
    plt.title("Mean of input")
    plt.xlabel("Count")
    plt.ylabel("Mean")
    plt.xticks(x)

    plt.subplot(3, 1, 2)
    x = []
    y = []
    for key, values in stds.items():
        x.extend([key] * len(values))
        y.extend(values)
    plt.scatter(x, y, color="blue", alpha=alpha)
    plt.title("STD of input")
    plt.xlabel("Count")
    plt.ylabel("STD")
    plt.xticks(x)

    plt.subplot(3, 1, 3)
    x = []
    y = []
    for key, values in gmdbs.items():
        x.extend([key] * len(values))
        y.extend(values)
    plt.scatter(x, y, color="blue", alpha=alpha)
    plt.title("Relative values")
    plt.xlabel("Count")
    plt.ylabel("Mean / STD")
    plt.xticks(x)

    plt.tight_layout()
    plt.show()

