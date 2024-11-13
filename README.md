# Occupancy estimation

## Introduction

## Installation

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
