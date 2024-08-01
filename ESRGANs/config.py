import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET = "../../../Datasets/DIV-2K"
GEN_CHECKPOINT = 'SRGANs/checkpoints/gen_checkpoint.pth.tar'
DISC_CHECKPOINT = 'SRGANs/checkpoints/disc_checkpoint.pth.tar'
SAVE_MODEL = True
LOAD_MODEL = False
EPOCHS = 100000
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
HIGH_RES = 128
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3
LAMBDA_GP = 10
NUM_WORKERS = 4


highres_transform = A.Compose([
    A.Normalize([0 for _ in range(IMG_CHANNELS)], [1 for _ in range(IMG_CHANNELS)]),
    ToTensorV2()
])

lowres_transform = A.Compose([
    A.Resize(height=LOW_RES, width=LOW_RES, interpolation=Image.BICUBIC),
    A.Normalize([0 for _ in range(IMG_CHANNELS)], [1 for _ in range(IMG_CHANNELS)]),
    ToTensorV2()
])

both_transform = A.Compose([
    A.RandomCrop(height=HIGH_RES, width=HIGH_RES),
    A.HorizontalFlip(0.5),
    A.RandomRotate90(0.5)
])