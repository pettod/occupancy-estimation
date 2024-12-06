# Occupancy estimation

## Introduction

## Installation

Install Nvidia drivers if not there:

```bash
sudo apt update
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers devices
sudo apt install nvidia-driver-xxx
```

Install anaconda (if not installed):

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
```

Answer "yes":
You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[no] >>> yes

### Option 1

```bash
conda create -n mems python=3.10.15
conda activate mems
conda install pytorch::pytorch torchvision torchaudio -c pytorch
sudo apt install portaudio19-dev
pip install -e .
```

If pyaudio issues:

```bash
sudo apt install portaudio19-dev
```

### Option 2

```bash
conda env create -f environment.yml
conda activate mems
```

## Train

```bash
python train.py <name_of_the_training>
```

## TODO

- Use buckets: 0, 1-5, 6-10, 11-15, 16-20, 21-25, 26-30, 31-35, 36-40, 41-45, 46-50
- Augment data: transforms.FrequencyMasking, transforms.TimeMasking, https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
- use desibels
  def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)
- use STFT

## Didin't work
- log of mel values (new mean and std)
