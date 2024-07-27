import torch
from math import log2

START_IMG_SIZE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZES = [16, 16, 16, 16, 16, 16, 16, 8, 4]
DATASET = "Downloads//Datasets//celeba_hq"
GEN_CHECKPOINT = 'ProGANS//checkpoints//gen_checkpoint.pth.tar'
CRITIC_CHECKPOINT = 'ProGANS//checkpoints//critic_checkpoint.pth.tar'
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
IMG_SIZE = 1024
IMG_CHANNELS = 3
Z_DIM = 256
IN_CHANNELS = 256
LAMBDA_GP = 10
NUM_STEPS = int(log2(IMG_SIZE / 4)) + 1
PROGRESSIVE_EPOCHS = [10] * len(BATCH_SIZES)
FIXED_NOISE = torch.rand((1, Z_DIM, 1, 1)).to(DEVICE)
NUM_WORKERS = 4
