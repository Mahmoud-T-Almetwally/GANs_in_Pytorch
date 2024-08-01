import torch
from torch import nn
from torch import optim
from utils import save_checkpoint, load_checkpoint, plot_examples, gradient_penalty
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
import config
from dataset import MyImageFolder
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = True

def train_step(gen, disc, gen_opt, disc_opt, gen_scalar, disc_scalar, MSE, L1, VGGLoss, loader):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        low_res = low_res.to(config.DEVICE)
        high_res = high_res.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(low_res)
            real_disc = disc(high_res)
            fake_disc = disc(fake.detach())
            gp = gradient_penalty(disc, high_res, fake, device=config.DEVICE)
            disc_loss = - torch.mean(real_disc) - torch.mean(fake_disc) + config.LAMBDA_GP * gp

        disc_opt.zero_grad()
        disc_scalar.scale(disc_loss).backward()
        disc_scalar.step(disc_opt)
        disc_scalar.update()

        with torch.cuda.amp.autocast():
            l1 = 1e-2 * L1(fake, high_res)
            adverserial_loss = 1e-5 * (-torch.mean(disc(fake)))
            vgg_loss = VGGLoss(fake, high_res)
            gen_loss = l1 + adverserial_loss + vgg_loss

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
    l1 = nn.L1Loss()
    vgg_loss = VGGLoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.DISC_CHECKPOINT, disc, disc_opt, config.LEARNING_RATE)
        load_checkpoint(config.GEN_CHECKPOINT, gen, gen_opt, config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        train_step(gen, disc, gen_opt, disc_opt, gen_scalar, disc_scalar, mse, l1, vgg_loss, loader)
        
        if config.SAVE_MODEL:
            save_checkpoint(gen, gen_opt, config.GEN_CHECKPOINT)
            save_checkpoint(disc, disc_opt, config.DISC_CHECKPOINT)

if __name__ == '__main__':
    main()