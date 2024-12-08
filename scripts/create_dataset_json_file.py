import pandas as pd
import torchaudio
import json
import os
from datetime import datetime
from glob import glob
from tqdm import tqdm
import random


DATA_ROOT = "data_original"
NEW_DATA_ROOT = "data_high-pass"  # None
SAMPLE_LENGTH = 60.0
TRAIN_VALID_SPLIT = 0.8
PRINT_JSON = False
USE_BUCKETS = False
BUCKETS = [0, 3, 7, 12, 19, 27, 38, 50]


def replaceRootPath(original_path, new_root):
    path_parts = original_path.split(os.sep)
    transformed_path = os.path.join(new_root, *path_parts[1:])
    return transformed_path


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


def getBucket(count):
    try:
        for i, bucket in enumerate(BUCKETS):
            if count <= bucket:
                return i
    except Exception as e:
        raise ValueError(
            f"Too high count: {count}, bucket max is {BUCKETS[-1]}")


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
        if NEW_DATA_ROOT:
            audio_file_path = replaceRootPath(audio_file_path, NEW_DATA_ROOT)

        # Loop CSV file timestamps
        for csv_index, row in csv_info:
            csv_start_timestamp = row["Seconds"]
            count_time = 365*24*3600  # Big initial value
            if csv_index < len(csv_info) - 1:
                next_row = csv_info[csv_index + 1][1]
                count_time = next_row["Seconds"] - csv_start_timestamp
            csv_end_timestamp = csv_start_timestamp + count_time

            # Timestamp in audio file
            if (csv_end_timestamp >= audio_file_start_time and
                csv_start_timestamp <= audio_file_end_time - sample_length
            ):
                
                count = row["Count"]
                if USE_BUCKETS:
                    count = getBucket(count)
                start_time = max(0.0, csv_start_timestamp - audio_file_start_time)
                max_audio_file_count_time = audio_file_length - start_time
                if (count_time is None or
                    count_time > max_audio_file_count_time
                ):
                    count_time = max_audio_file_count_time
                count_end_time = count_time + start_time
                while start_time + sample_length <= count_end_time:
                    end_time = start_time + sample_length
                    single_sample = {
                        "audio_file_path": audio_file_path,
                        "count": count,
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_timestamp": datetime.fromtimestamp(
                            audio_file_start_time + start_time).strftime(
                            "%Y-%m-%d %H:%M:%S"),
                    }
                    if USE_BUCKETS:
                        single_sample["buckets"] = BUCKETS
                    samples.append(single_sample)
                    start_time += sample_length
    return samples


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"dataset_{str(SAMPLE_LENGTH).zfill(2).replace('.', '-')}s_{timestamp}"
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
