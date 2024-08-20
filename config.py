import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
import albumentations as A

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# TRAIN_DIR = 'pix2pix-dataset/edges2shoes/edges2shoes/train'
# VAL_DIR = 'pix2pix-dataset/edges2shoes/edges2shoes/val'
LEARNING_RATE = 2e-4
BATCH_SIZE = 24
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 100
NUM_EPOCHS = 351
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC =  'Pix2Pix_Buildings/Model/_512/disc.pth.tar'
CHECKPOINT_GEN = 'Pix2Pix_Buildings/Model_512/gen.pth.tar'

both_transform = A.Compose(
    [A.Resize(width = 512, height = 512)], additional_targets={'image0':'image'}
)

transform = A.Compose(
    [
        A.Resize(width = 512, height = 512),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2()
    ]
)

transform_512 = A.Compose(
    [
        A.Resize(width = 512, height = 512),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2()
    ]
)

transform_only_input = A.Compose(
    [
    # A.HorizontalFlip(p = 0.5),
    # A.ColorJitter(p = 0.2),
    A.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], max_pixel_value=255.0),
    ToTensorV2()
    ]
)

transform_only_mask = A.Compose([
    A.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
    ToTensorV2()
])