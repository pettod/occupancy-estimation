import csv
from datetime import datetime
import os


WAV_FILE_PATH = "/Users/todorov/Documents/ef/count-ml/20241016_155159.WAV"
COUNT = WAV_FILE_PATH.split("/")[-2].split("_")[0]


def extractAndFormatTimestamp(filename):
    if filename.endswith(".WAV"):
        filename = filename[:-4]
    date_str, time_str = os.path.basename(filename).split("_")  # "YYYYMMDD_HHMMSS"
    timestamp = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    unix_seconds = float(timestamp.timestamp())
    formatted_time = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
    return formatted_time, unix_seconds


def writeToCsv(wav_filename):
    csv_filename = os.path.splitext(wav_filename)[0] + ".csv"
    formatted_time, unix_seconds = extractAndFormatTimestamp(wav_filename)
    data_to_add = [
        ["Time", "Seconds", "Seconds-from-start", "Count"],
        [formatted_time, f"{unix_seconds:.2f}", 0.0, COUNT]
    ]
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data_to_add)    
    print(f"CSV file created: '{csv_filename}'")


if __name__ == "__main__":
    writeToCsv(WAV_FILE_PATH)
