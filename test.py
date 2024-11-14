import numpy as np
import os
import torch
from importlib import import_module
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Project files
from src.dataset import AudioSpectrogramDataset as Dataset
from src.utils.utils import loadModel


# Data paths
DATA_FILENAME = "dataset_2-0s_valid.json"
REPLACED_DATA_PATH_ROOT = "data_high-pass"

# Model parameters
MODEL_PATHS = [
    "saved_models/2024-11-14_123650_18k_samples",
]
BATCH_SIZE = 16
DEVICE = torch.device("cpu")


def loadModels():
    models = []
    for model_path in MODEL_PATHS:
        config = import_module(os.path.join(
            model_path, "codes.config").replace("/", ".")).CONFIG
        model = config.MODELS[0].to(DEVICE)
        loadModel(model, model_path=model_path)
        models.append(model)
    return models


def predictResults():
    # Dataset
    dataset = Dataset(DATA_FILENAME, REPLACED_DATA_PATH_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Save directory
    with torch.no_grad():
        model = loadModels()[0]
        gt_pred = {}

        # Predict and save
        for i, (x, y) in enumerate(tqdm(dataloader)):
            x, y = x.to(DEVICE), y.numpy()
            predictions = model(x).squeeze(1).cpu().numpy()
            for gt, pred in zip(y, predictions):
                gt = gt.astype(int)
                pred = np.round(pred).astype(int)
                if gt in gt_pred:
                    gt_pred[gt].append(pred)
                else:
                    gt_pred[gt] = [pred]
            #if i == 10:
            #    break
    return gt_pred


def main():
    results = predictResults()

    # Prepare data for visualization
    keys = sorted(results.keys())
    means = []
    standard_deviations = []
    min_values = []
    max_values = []

    # Calculate standard deviation, minimum, and maximum for each list
    for key, values in results.items():
        mean = np.mean(values)
        std_dev = np.std(values)
        means.append(mean)
        standard_deviations.append(std_dev)
        min_values.append(min(values))
        max_values.append(max(values))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, key in enumerate(keys):
        ax.plot([i, i], [means[i] - standard_deviations[i], means[i] + standard_deviations[i]], color="red", linewidth=4, alpha=1.0, label="Standard deviation" if i == 0 else "")
        ax.plot([i, i], [min_values[i], max_values[i]], color="black", linewidth=1, alpha=1.0, label="Min and max" if i == 0 else "")
        ax.plot(i, max_values[i], "_", color="black", markersize=6)
        ax.plot(i, min_values[i], "_", color="black", markersize=6)
        ax.plot(i, means[i], "o", color="green", markersize=6, label="Prediction mean" if i == 0 else "")
        ax.plot(i, key, "o", color="blue", markersize=6, label="Ground truth" if i == 0 else "")

    # Customize the plot
    ax.set_xlabel("Occupancy")
    ax.set_ylabel("Prediction")
    ax.set_title("Occupancy deviation")
    plt.xticks(ticks=range(len(keys)), labels=keys)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # Show the plot
    plt.show()


main()
