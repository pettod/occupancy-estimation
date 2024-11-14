import pandas as pd
import torchaudio
import json
import os
from datetime import datetime
from glob import glob
from tqdm import tqdm
import random


DATA_ROOT = "data_original"
SAMPLE_LENGTH = 2.0
TRAIN_VALID_SPLIT = 0.8
PRINT_JSON = False


def convertToUnixSeconds(timestamp):
    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    epoch = datetime(1970, 1, 1)
    total_seconds = int((dt - epoch).total_seconds())    
    return total_seconds


def audioLength(filepath):
    waveform, sample_rate = torchaudio.load(filepath)
    num_samples = waveform.size(1)
    duration_seconds = num_samples / sample_rate
    return duration_seconds


def getSamples(folder, sample_length=2.0):
    # Read data
    audio_file_paths = sorted(glob(f"{folder}/*.WAV"))
    csv_file_path = glob(f"{folder}/*.csv")
    if len(csv_file_path) != 1:
        raise Exception(f"Expected 1 csv file, found {len(csv_file_path)}")
    df = pd.read_csv(csv_file_path[0], delimiter=",")

    # Loop audio files
    samples = []
    csv_info = list(df.iterrows())
    for i, audio_file_path in enumerate(audio_file_paths):

        # Audio details
        audio_file_name = audio_file_path.split("/")[-1]
        audio_file_start_time = convertToUnixSeconds(audio_file_name[0:15])
        audio_file_length = audioLength(audio_file_path)
        audio_file_end_time = audio_file_start_time + audio_file_length

        # Loop CSV file timestamps
        for csv_index, row in csv_info:
            csv_start_timestamp = row["Seconds"]
            occupancy_time = 365*24*3600  # Big initial value
            if csv_index < len(csv_info) - 1:
                next_row = csv_info[csv_index + 1][1]
                occupancy_time = next_row["Seconds"] - csv_start_timestamp
            csv_end_timestamp = csv_start_timestamp + occupancy_time

            # Timestamp in audio file
            if (csv_end_timestamp >= audio_file_start_time and
                csv_start_timestamp <= audio_file_end_time - sample_length
            ):
                
                occupancy = row["Count"]
                start_time = max(0.0, csv_start_timestamp - audio_file_start_time)
                max_audio_file_occupancy_time = audio_file_length - start_time
                if (occupancy_time is None or
                    occupancy_time > max_audio_file_occupancy_time
                ):
                    occupancy_time = max_audio_file_occupancy_time
                occupancy_end_time = occupancy_time + start_time
                while start_time + sample_length <= occupancy_end_time:
                    end_time = start_time + sample_length
                    single_sample = {
                        "audio_file_path": audio_file_path,
                        "occupancy": occupancy,
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_timestamp": datetime.fromtimestamp(
                            audio_file_start_time + start_time).strftime(
                            "%Y-%m-%d %H:%M:%S"),
                    }
                    samples.append(single_sample)
                    start_time += sample_length
    return samples


def main():
    json_filename = f"dataset_{str(SAMPLE_LENGTH).replace('.', '-')}s"
    data_folders = glob(os.path.join(DATA_ROOT, "*/*/*/"))
    train_samples = []
    valid_samples = []
    for data_folder in tqdm(data_folders):
        samples_per_recording_session = getSamples(data_folder, SAMPLE_LENGTH)
        for sample in samples_per_recording_session:
            if random.random() < TRAIN_VALID_SPLIT:
                train_samples.append(sample)
            else:
                valid_samples.append(sample)
    if PRINT_JSON:
        for sample in train_samples:
            print("{")
            for key, value in sample.items():
                print(f"    {key}: {value}")
            print("},")
    else:
        json_filename_train = f"{json_filename}_train.json"
        json_filename_valid = f"{json_filename}_valid.json"
        with open(json_filename_train, "w") as json_file:
            json.dump(train_samples, json_file, indent=4)
        with open(json_filename_valid, "w") as json_file:
            json.dump(valid_samples, json_file, indent=4)
        print("File saved:", json_filename_train)
        print("File saved:", json_filename_valid)
    print(len(train_samples), "train samples")
    print(len(valid_samples), "valid samples")
    print(len(train_samples) + len(valid_samples), "total samples")


main()
