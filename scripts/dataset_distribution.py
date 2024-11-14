import json
import matplotlib.pyplot as plt


train_file = "dataset_2-0s_train.json"
valid_file = "dataset_2-0s_valid.json"

# Step 1: Read the JSON data
with open(train_file, "r") as file:
    data = json.load(file)
with open(valid_file, "r") as file:
    data += json.load(file)
occupancies = [entry["occupancy"] for entry in data]

# Step 3: Plot the histogram of counts
bins = range(min(occupancies), max(occupancies) + 2)
hist_values, bins, _ = plt.hist(occupancies, bins=bins, edgecolor="black")

# Step 4: Center-align ticks by setting them to the middle of each bin
bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
plt.xticks(bin_centers, labels=[str(int(bin)) for bin in bins[:-1]])

plt.xlabel("Occupancy count")
plt.ylabel("Number of samples")
plt.title("Occupancy distribution")

# Show the plot
plt.show()
