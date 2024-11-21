import numpy as np
import os
import torch
from importlib import import_module
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import cpu_count

# Project files
from src.dataset import AudioSpectrogramDataset as Dataset
from src.utils.utils import loadModel


# Data paths
DATA_FILENAME = "dataset_60-0s_train.json"
REPLACED_DATA_PATH_ROOT = "data_high-pass"

# Model parameters
MODEL_PATH = "saved_models/2024-11-20_110334_60s-window_1500-samples"
BATCH_SIZE = 16
DEVICE = torch.device("mps")
NUM_WORKERS = 0  #cpu_count()

# Additional
PRINT_AUDIO_THRESHOLD = 0.1  # Set to 0, if you don't want info


def predictResults():
    # Save directory
    with torch.no_grad():
        model, config = loadModelAndConfig()

        # Dataset
        sample_rate = config.SAMPLE_RATE
        transform = config.TRANSFORM
        input_normalize = config.INPUT_NORMALIZE
        dataset = Dataset(DATA_FILENAME, REPLACED_DATA_PATH_ROOT, transform, input_normalize, sample_rate, True)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        gt_pred = {}

        # Predict and save
        for i, (x, y, file_info) in enumerate(tqdm(dataloader)):
            #plt.imshow(x[0].squeeze(0).numpy(), aspect='auto', origin='lower')
            #plt.specgram(x[0].squeeze(0).numpy(), NFFT=1024, Fs=sample_rate, noverlap=512, cmap='viridis')
            #plt.plot(x[0].squeeze(0).numpy())
            #plt.title(y[0])
            #plt.show()
            x, y = x.to(DEVICE), y.numpy()
            predictions = model(x).squeeze(1).cpu().numpy()
            for i, (gt, pred) in enumerate(zip(y, predictions)):
                gt = gt.astype(int)
                if len(pred.shape) == 0:
                    pred = np.round(pred).astype(int)
                else:
                    pred = np.argmax(pred).astype(int)
                if gt in gt_pred:
                    gt_pred[gt].append(pred)
                else:
                    gt_pred[gt] = [pred]
                pred_accuracy = accuracy(pred, gt)
                if pred_accuracy < PRINT_AUDIO_THRESHOLD:
                    print("\nOccupancy: ", gt)
                    print("Prediction:", pred)
                    print("Accuracy:   {}%".format(round(100 * pred_accuracy, 1)))
                    print("Original file path: ", file_info["audio_file_path"][i])
                    print("Replaced file path: ", file_info["replaced_audio_file_path"][i])
                    start = int(file_info["start_time"][i])
                    end = int(file_info["end_time"][i])
                    print(f"Time: {start//60:02}:{start%60:02} - {end//60:02}:{end%60:02}\n")
            #if i == 10:
            #    break
    return gt_pred


def loadModelAndConfig():
    config = import_module(os.path.join(
        MODEL_PATH, "codes.config").replace("/", ".")).CONFIG
    model = config.MODELS[0].to(DEVICE)
    loadModel(model, model_path=MODEL_PATH)
    return model, config


def accuracy(pred, gt):
    if pred == 0 or gt == 0:
        return 0.0
    else:
        return min(pred, gt) / max(pred, gt)


def accuracyAndStdDataset(gt_pred):
    accuracies = []
    absolute_stds = []
    relative_stds = []
    for gt in gt_pred.keys():
        predictions = gt_pred[gt]
        accuracies += [accuracy(p, gt) for p in predictions]
        std = np.std(predictions)
        absolute_stds.append(std)
        if gt != 0:
            relative_stds.append(std / gt)
    return np.mean(accuracies), np.mean(absolute_stds), np.mean(relative_stds)


def candleChart(gt_pred, marker_width=8, alpha=0.5):
    # Plot data
    fig, ax = plt.subplots(figsize=(8, 6))
    gt_occupancies = sorted(gt_pred.keys())
    pred_occupancies =[]
    for i, key in enumerate(gt_occupancies):
        values = gt_pred[key]
        mean = np.mean(values)
        pred_occupancies.append(mean)
        std = np.std(values)
        min_value = min(values)
        max_value = max(values)
        ax.plot([i, i], [mean - std, mean + std], color="green", linewidth=marker_width, alpha=alpha, label="Standard deviation" if i == 0 else "")
        ax.plot([i, i], [min_value, max_value], color="black", linewidth=1, alpha=1.0, label="Min and max" if i == 0 else "")
        ax.plot(i, key, "o", color="blue", markersize=marker_width, label="Ground truth" if i == 0 else "")
        ax.plot(i, max_value, "_", color="black", markersize=marker_width)
        ax.plot(i, min_value, "_", color="black", markersize=marker_width)
        ax.plot(i, mean, "x", color="red", markersize=marker_width, label="Prediction mean" if i == 0 else "")
    ax.plot(range(len(gt_occupancies)), gt_occupancies, color="blue", alpha=alpha)
    ax.plot(range(len(gt_occupancies)), pred_occupancies, color="red", alpha=alpha)

    # Plot style
    pred_accuracy, pred_absolute_std, pred_relative_std = accuracyAndStdDataset(gt_pred)
    ax.set_xlabel("Ground truth occupancy")
    ax.set_ylabel("Predicted occupancy")
    ax.set_title("Occupancy prediction\naccuracy: {}%, absolute STD: {} people, relative STD: {}%".format(
        round(100 * pred_accuracy, 1),
        round(pred_absolute_std, 1),
        round(100 * pred_relative_std, 1)))
    ax.set_ylim(0)
    plt.xticks(ticks=range(len(gt_occupancies)), labels=gt_occupancies)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def getExampleData():
    gt_pred = {}
    for gt in range(40):
        number_of_samples = 20 + np.random.randint(20, size=1)[0]
        low_value = np.random.randint(-10, -1)
        high_value = np.random.randint(1, 10)
        predicted_occupancies = np.array(gt) + np.random.randint(low_value, high_value, size=number_of_samples)
        predicted_occupancies[predicted_occupancies < 0] = 0
        gt_pred[gt] = list(predicted_occupancies)
    return gt_pred


if __name__ == "__main__":
    gt_pred = predictResults()  #getExampleData()
    candleChart(gt_pred)
