import numpy as np
import os
import torch
from importlib import import_module
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import cpu_count
import sys
import datetime

# Project files
from src.dataset import AudioSpectrogramDataset as Dataset
from src.utils.utils import loadModel


# Data paths
DATA_FILENAME = "dataset_60-0s_train.json"
REPLACED_DATA_PATH_ROOT = "data_high-pass"

# Model parameters
MODEL_PATH = "saved_models/2024-11-24_162635_32kHz-sampling-rate"
MODEL_FILE_NAME = "model_0.pt" # model_0.pt or last_model_0.pt
BATCH_SIZE = 16
DEVICE = torch.device("mps")
NUM_WORKERS = 0  #cpu_count()

# Additional
PRINT_AUDIO_THRESHOLD = 0.1  # Set to 0, if you don't want info
PLOT_RESULT = True if len(sys.argv) == 1 else False


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
        for j, (x, y, file_info) in enumerate(tqdm(dataloader)):
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
            #if j == 0:
            #    break
    return gt_pred


def loadModelAndConfig():
    config = import_module(os.path.join(
        MODEL_PATH, "codes.config").replace("/", ".")).CONFIG
    model = config.MODELS[0]
    loadModel(model, model_path=MODEL_PATH, model_file_name=MODEL_FILE_NAME)
    model = model.to(DEVICE)
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
    fig, ax = plt.subplots(
        2, 1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [5, 1]},
        sharex=True,
    )
    plt.subplots_adjust(wspace=0, hspace=0)
    gt_occupancies = sorted(gt_pred.keys())
    number_of_occupancies = [len(gt_pred[gt]) for gt in gt_occupancies]

    # Plot precition accuracy candle chart
    pred_occupancies =[]
    for i, key in enumerate(gt_occupancies):
        values = gt_pred[key]
        mean = np.mean(values)
        pred_occupancies.append(mean)
        std = np.std(values)
        min_value = min(values)
        max_value = max(values)
        ax[0].plot([i, i], [mean - std, mean + std], color="green", linewidth=marker_width, alpha=alpha, label="Standard deviation" if i == 0 else "")
        ax[0].plot([i, i], [min_value, max_value], color="black", linewidth=1, alpha=1.0, label="Min and max" if i == 0 else "")
        ax[0].plot(i, key, "o", color="blue", markersize=marker_width, label="Ground truth" if i == 0 else "")
        ax[0].plot(i, max_value, "_", color="black", markersize=marker_width)
        ax[0].plot(i, min_value, "_", color="black", markersize=marker_width)
        ax[0].plot(i, mean, "x", color="red", markersize=marker_width, label="Prediction mean" if i == 0 else "")
    ax[0].plot(range(len(gt_occupancies)), gt_occupancies, color="blue", alpha=alpha)
    ax[0].plot(range(len(gt_occupancies)), pred_occupancies, color="red", alpha=alpha)

    # Plot number of samples histogram
    bins = range(len(number_of_occupancies))
    ax[1].bar(bins, number_of_occupancies, edgecolor="black", align="center", width=1.0)

    # Add text on top of bins
    for x, value in zip(bins, number_of_occupancies):
        ax[1].text(
            x,
            value, str(int(value)),
            ha="center", va="bottom", fontsize=5,
        )

    # Plot style
    pred_accuracy, pred_absolute_std, pred_relative_std = accuracyAndStdDataset(gt_pred)
    ax[0].set_title(
        f"Occupancy Prediction\n"
        f"$\mathbf{{Accuracy:}}$ {round(100 * pred_accuracy, 1)}%, "
        f"$\mathbf{{Absolute\ STD:}}$ {round(pred_absolute_std, 1)} people, "
        f"$\mathbf{{Relative\ STD:}}$ {round(100 * pred_relative_std, 1)}%",
        loc="center",
    )
    ax[0].set_ylabel("Predicted occupancy")
    ax[0].set_ylim(0)
    plt.xticks(ticks=range(len(gt_occupancies)), labels=gt_occupancies)
    ax[0].legend()
    ax[0].grid(True, linestyle="--", alpha=0.5)
    ax[1].set_ylim(0, max(number_of_occupancies) * 1.2)
    ax[1].set_xlabel("Ground truth occupancy")
    ax[1].set_ylabel("Samples")
    if PLOT_RESULT:
        plt.show()    
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        plt.savefig(f"{timestamp}_candle_chart.png", dpi=300)


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
    gt_pred = predictResults() if True else getExampleData()
    candleChart(gt_pred)
