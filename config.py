import os
from multiprocessing import cpu_count

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import (
    Compose, ToTensor, Normalize
)

from src.dataset import AudioSpectrogramDataset as Dataset
from src.loss_functions import l1, GANLoss
from src.architectures.discriminator import UNetDiscriminatorSN
from src.architectures.model import Net


DATA_ROOT = os.path.realpath(".")


class CONFIG:
    # Data paths
    TRAIN_FILE = os.path.join(DATA_ROOT, "dataset_2-0s.json")
    VALID_FILE = os.path.join(DATA_ROOT, "dataset_2-0s.json")
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
    TRAIN_TRANSFORM = None
    VALID_TRANSFORM = None
    TEST_TRANSFORM = ToTensor()
    INPUT_NORMALIZE = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    TRAIN_DATASET = Dataset(TRAIN_FILE)
    VALID_DATASET = Dataset(VALID_FILE)

    # General parameters
    DROP_LAST_BATCH = False
    NUMBER_OF_DATALOADER_WORKERS = 0 #cpu_count()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
