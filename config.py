import os
from multiprocessing import cpu_count

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Normalize
import torchaudio.transforms as T

from src.dataset import AudioSpectrogramDataset as Dataset
from src.loss_functions import l1, GANLoss, crossEntropy
from src.architectures.discriminator import UNetDiscriminatorSN
from src.architectures.model import Net


class CONFIG:
    # Data paths
    TRAIN_FILE = "dataset_2-0s_train.json"
    VALID_FILE = "dataset_2-0s_valid.json"
    REPLACED_DATA_PATH_ROOT = "data_high-pass"
    TEST_IMAGE_PATHS = [
    ]

    # Model
    SEED = 5432367
    MODELS = [
        Net(),
    ]
    OPTIMIZERS = [
        optim.Adam(MODELS[0].parameters(), lr=1e-4),
    ]
    SCHEDULERS = [
        ReduceLROnPlateau(OPTIMIZERS[0], "min", 0.3, 6, min_lr=1e-8),
    ]

    # Model loading
    LOAD_MODELS = [
        False,
    ]
    MODEL_PATHS = [
        None,
    ]
    LOAD_OPTIMIZER_STATES = [
        False,
    ]
    CREATE_NEW_MODEL_DIR = True

    # Cost function
    LOSS_FUNCTIONS = [
        [l1],
    ]
    LOSS_WEIGHTS = [
        [1],
    ]

    # Hyperparameters
    EPOCHS = 1000
    BATCH_SIZE = 16
    PATIENCE = 6
    ITERATIONS_PER_EPOCH = 1

    # Transforms and dataset
    DATA_MEAN = 203.9879
    DATA_STD = 0.7629
    SAMPLE_RATE = 192000
    TRANSFORM = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=256)
    INPUT_NORMALIZE = Normalize([DATA_MEAN], [DATA_STD])
    TRAIN_DATASET = Dataset(TRAIN_FILE, REPLACED_DATA_PATH_ROOT, TRANSFORM, INPUT_NORMALIZE, SAMPLE_RATE)
    VALID_DATASET = Dataset(VALID_FILE, REPLACED_DATA_PATH_ROOT, TRANSFORM, INPUT_NORMALIZE, SAMPLE_RATE)

    # General parameters
    DROP_LAST_BATCH = False
    NUMBER_OF_DATALOADER_WORKERS = cpu_count()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # GAN
    USE_GAN = False
    DISCRIMINATOR = UNetDiscriminatorSN(3)
    DIS_OPTIMIZER = optim.Adam(DISCRIMINATOR.parameters(), lr=1e-4)
    DIS_SCHEDULER = ReduceLROnPlateau(DIS_OPTIMIZER, "min", 0.3, 6, min_lr=1e-8)
    DIS_LOSS = GANLoss("vanilla")
    DIS_LOSS_WEIGHT = 1

    # Load GAN
    LOAD_GAN = False
    DIS_PATH = None
    LOAD_DIS_OPTIMIZER_STATE = False
