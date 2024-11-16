# Occupancy estimation

## Introduction

## Installation

Install anaconda:

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
```

Answer "yes":
You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[no] >>> yes

### Option 1

```bash
conda env create -f environment.yml
conda activate mems
```

### Option 2

```bash
conda create -n mems python=3.10.15
conda activate mems
conda install pytorch::pytorch torchvision torchaudio -c pytorch
pip install -e .
```

## Train

```bash
python train.py <name_of_the_training>
```
