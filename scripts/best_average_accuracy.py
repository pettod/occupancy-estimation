import json
import sys
import numpy as np


def accuracy(pred, gt):
    if pred == 0 or gt == 0:
        return 0.0
    else:
        return min(pred, gt) / max(pred, gt)


def main():
    train_file = sys.argv[1]
    with open(train_file, "r") as f:
        train_data = json.load(f)
    occupancies = np.array([x["occupancy"] for x in train_data])
    best_occupancy = None
    best_accuracy = 0
    for pred in range(100):
        pred_accuracy = 0
        for gt in occupancies:
            pred_accuracy += accuracy(pred, gt)
        pred_accuracy /= len(occupancies)
        if pred_accuracy > best_accuracy:
            best_accuracy = pred_accuracy
            best_occupancy = pred
    print("Best accuracy: {}%".format(round(100 * best_accuracy, 2)))
    print("Best occupancy: {}".format(best_occupancy))



main()
