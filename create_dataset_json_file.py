from glob import glob
from datetime import datetime
import pandas as pd
import torchaudio
import json

SAMPLE_LENGTH = 2.0
JSON_FILE_NAME = "20241104_105000_cafe.json"
PRINT_JSON = True


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
    audio_file_paths = sorted(glob(f"{folder}/*.WAV"))
    csv_file_path = glob(f"{folder}/*.csv")
    if len(csv_file_path) != 1:
        raise Exception(f"Expected 1 csv file, found {len(csv_file_path)}")
    df = pd.read_csv(csv_file_path[0], delimiter=",")
    samples = []
    no_audio_available_for_count = False

    # Loop CSV file time stamps
    for csv_index, row in df.iterrows():
        csv_timestamp = row["Seconds"]

        # Find audio file which ending timestamp (starting timestamp + length)
        # is smaller than CSV time stamp
        for i, audio_file_path in enumerate(audio_file_paths):
            audio_file_name = audio_file_path.split("/")[-1]
            audio_file_start_time = convertToUnixSeconds(audio_file_name[0:15])
            audio_file_length = audioLength(audio_file_path)
            audio_file_end_time = audio_file_start_time + audio_file_length
            if (csv_timestamp > audio_file_start_time and
                csv_timestamp < audio_file_end_time - sample_length
            ):
                break
            elif i == len(audio_file_paths) - 1:
                print(f"Audio file not found for timestamp {csv_timestamp}")
                no_audio_available_for_count = True
                #raise Exception("No audio file found")

        if no_audio_available_for_count:
            break

        # Only one sample per window now
        # Fix this
        occupancy = row["Count"]
        start_time = max(0.0, csv_timestamp - audio_file_start_time)
        end_time = start_time + sample_length
        single_sample = {
            "audio_file_path": audio_file_path,
            "occupancy": occupancy,
            "start_time": start_time,
            "end_time": end_time,
        }
        samples.append(single_sample)
    return samples


def main():
    samples = getSamples("data/0_input/20241104_105000_cafe", SAMPLE_LENGTH)
    if PRINT_JSON:
        for sample in samples:
            print("{")
            for key, value in sample.items():
                print(f"    {key}: {value}")
            print("},")
    else:
        with open(JSON_FILE_NAME, "w") as json_file:
            json.dump(samples, json_file, indent=4)
        print("File saved:", JSON_FILE_NAME)


main()
