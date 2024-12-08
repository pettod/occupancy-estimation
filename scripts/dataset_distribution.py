import json
import matplotlib.pyplot as plt


train_file = "dataset_60-0s_train.json"
valid_file = "dataset_60-0s_valid.json"

# Read data
with open(train_file, "r") as file:
    data = json.load(file)
with open(valid_file, "r") as file:
    data += json.load(file)
counts = [entry["count"] for entry in data]

# Plot histogram of counts
print("Number of samples:", len(counts))
bins = range(min(counts), max(counts) + 5)
plt.hist(counts, bins=bins, edgecolor="black", align="left")

plt.xlabel("Count")
plt.ylabel("Number of samples")
plt.title("Count distribution")
plt.show()
