import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    save_checkpoint,
    load_checkpoint,
    generate_examples,
    plot_to_tensorboard
)
import config
from Model import Generator, Discriminator
from tqdm.auto import tqdm
from math import log2

torch.backends.cudnn.benchmark = True

def get_loader(img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize([0.5 for _ in range(config.IMG_CHANNELS)],
                             [0.5 for _ in range(config.IMG_CHANNELS)])
    ])
    batch_size = config.BATCH_SIZES[int(log2(img_size / 4))]
    dataset = datasets.ImageFolder(config.DATASET, transform=transform)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=config.NUM_WORKERS,
                        pin_memory=True)
    
    return loader, dataset

def train_step(gen, critic, gen_optim, critic_optim, gen_scalar, critic_scalar, step, alpha, writer, tensorboard_step, dataset, loader):
    loop = tqdm(loader)
    
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]

        noise = torch.rand((cur_batch_size, config.Z_DIM, 1, 1)).to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            critic_real = critic(real,alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step, config.DEVICE)
            critic_loss = (- (torch.mean(critic_real) - torch.mean(critic_fake)) + gp * config.LAMBDA_GP + (0.001 * torch.mean(critic_real) ** 2))

        critic_optim.zero_grad()
        critic_scalar.scale(critic_loss).backward()
        critic_scalar.step(critic_optim)
        critic_scalar.update()

        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            gen_loss = -torch.mean(gen_fake)

        gen_optim.zero_grad()
        gen_scalar.scale(gen_loss).backward()
        gen_scalar.step(gen_optim)
        gen_scalar.update()

        alpha += cur_batch_size / (len(dataset) * config.PROGRESSIVE_EPOCHS[step] * 0.5)
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fake = gen(config.FIXED_NOISE, alpha, step)
            plot_to_tensorboard(
                writer,
                gen_loss.item(),
                critic_loss.item(),
                real.detach(),
                fixed_fake.detach(),
                tensorboard_step
            )
            tensorboard_step += 1
    return tensorboard_step, alpha


def main():
    gen = Generator(config.Z_DIM, config.IN_CHANNELS, config.IMG_CHANNELS).to(config.DEVICE)
    critic = Discriminator(config.IN_CHANNELS, config.IMG_CHANNELS).to(config.DEVICE)

    gen_optim = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    critic_optim = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))

    critic_scalar = torch.cuda.amp.GradScaler()
    gen_scalar = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir='logs/ProGANs')

    if config.LOAD_MODEL:
        load_checkpoint(config.GEN_CHECKPOINT, gen, gen_optim, config.LEARNING_RATE)
        load_checkpoint(config.CRITIC_CHECKPOINT, critic, critic_optim, config.LEARNING_RATE)

    gen.train()
    critic.train()
    tensorboard_step = 0
    step = int(log2(config.START_IMG_SIZE / 4))
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        loader, dataset = get_loader(4*2**step)
        print(f'Current Image Size {4*2**step}')

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            tensorboard_step, alpha = train_step(
                gen,
                critic,
                gen_optim,
                critic_optim,
                gen_scalar,
                critic_scalar,
                step, 
                alpha,
                writer,
                tensorboard_step,
                dataset,
                loader
            )
        
        if config.SAVE_MODEL:
            save_checkpoint(gen, gen_optim, config.GEN_CHECKPOINT)
            save_checkpoint(critic, critic_optim, config.CRITIC_CHECKPOINT)
        step += 1
        
    generate_examples(gen, step)

if __name__ == '__main__':
    main()