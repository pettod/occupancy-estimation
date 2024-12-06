from multiprocessing import cpu_count
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Normalize
import torchaudio.transforms as T

from src.dataset import AudioSpectrogramDataset as Dataset
from src.loss_functions import l1, GANLoss, crossEntropy, mseLossForClassification
from src.architectures.discriminator import UNetDiscriminatorSN
from src.architectures.model import Net


DATA_ROOT = ""


class CONFIG:
    # Data paths
    TRAIN_FILE = os.path.join(DATA_ROOT, "dataset_60-0s_train.json")
    VALID_FILE = os.path.join(DATA_ROOT, "dataset_60-0s_valid.json")
    REPLACED_DATA_PATH_ROOT = os.path.join(DATA_ROOT, "data_high-pass")
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
        [crossEntropy],
    ]
    LOSS_WEIGHTS = [
        [1],
    ]

    # Hyperparameters
    EPOCHS = 1000
    BATCH_SIZE = 16
    PATIENCE = 20
    ITERATIONS_PER_EPOCH = 1

    # Transforms and dataset
    DATA_MEAN = 14.4809
    DATA_STD = 552.4592
    SAMPLE_RATE = 192000
    TRANSFORM = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64, n_fft=8000)
    INPUT_NORMALIZE = Normalize([DATA_MEAN], [DATA_STD])
    TRAIN_DATASET = Dataset(TRAIN_FILE, REPLACED_DATA_PATH_ROOT, TRANSFORM, INPUT_NORMALIZE, SAMPLE_RATE, augment=True)
    VALID_DATASET = Dataset(VALID_FILE, REPLACED_DATA_PATH_ROOT, TRANSFORM, INPUT_NORMALIZE, SAMPLE_RATE, augment=False)

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
