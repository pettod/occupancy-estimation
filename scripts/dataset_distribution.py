import json
import matplotlib.pyplot as plt


train_file = "dataset_2-0s_train.json"
valid_file = "dataset_2-0s_valid.json"

# Read data
with open(train_file, "r") as file:
    data = json.load(file)
with open(valid_file, "r") as file:
    data += json.load(file)
occupancies = [entry["occupancy"] for entry in data]

# Plot histogram of counts
bins = range(min(occupancies), max(occupancies) + 5)
plt.hist(occupancies, bins=bins, edgecolor="black", align="left")

plt.xlabel("Occupancy count")
plt.ylabel("Number of samples")
plt.title("Occupancy distribution")
plt.show()
