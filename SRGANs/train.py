import torch
from torch import nn
from torch import optim
from utils import save_checkpoint, load_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
import config
from dataset import MyImageFolder
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = True

def train_step(gen, disc, gen_opt, disc_opt, gen_scalar, disc_scalar, MSE, BCE, VGGLoss, loader):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        low_res = low_res.to(config.DEVICE)
        high_res = high_res.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake_high_res = gen(low_res)
            real_disc = disc(high_res)
            fake_disc = disc(fake_high_res.detach())
            real_loss = BCE(real_disc, torch.ones_like(real_disc) - 0.1 * torch.rand_like(real_disc))
            fake_loss = BCE(fake_disc, torch.zeros_like(fake_disc))
            disc_loss = (real_loss + fake_loss) / 2.0

        disc_opt.zero_grad()
        disc_scalar.scale(disc_loss).backward()
        disc_scalar.step(disc_opt)
        disc_scalar.update()

        with torch.cuda.amp.autocast():
            outputs = disc(fake_high_res)
            #MSE_loss = MSE(fake_high_res, high_res)
            VGG_loss = 0.006 * VGGLoss(fake_high_res, high_res)
            adversarial_loss = 1e-3 * BCE(outputs, torch.ones_like(outputs))
            gen_loss = VGG_loss + adversarial_loss

        gen_opt.zero_grad()
        gen_scalar.scale(gen_loss).backward()
        gen_scalar.step(gen_opt)
        gen_scalar.update()

def main():
    dataset = MyImageFolder(config.DATASET)
    loader = DataLoader(dataset, config.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=config.NUM_WORKERS)

    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(img_channels=3).to(config.DEVICE)
    gen_opt = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE)
    disc_opt = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE)
    gen_scalar = torch.cuda.amp.GradScaler()
    disc_scalar = torch.cuda.amp.GradScaler()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.DISC_CHECKPOINT, disc, disc_opt, config.LEARNING_RATE)
        load_checkpoint(config.GEN_CHECKPOINT, gen, gen_opt, config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        train_step(gen, disc, gen_opt, disc_opt, gen_scalar, disc_scalar, mse, bce, vgg_loss, loader)
        
        if config.SAVE_MODEL:
            save_checkpoint(gen, gen_opt, config.GEN_CHECKPOINT)
            save_checkpoint(disc, disc_opt, config.DISC_CHECKPOINT)

if __name__ == '__main__':
    main()