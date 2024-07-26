import torch
import random
import numpy as np
import os
import torchvision
import config
from torchvision.utils import save_image, make_grid

def plot_to_tensorboard(writer, Loss_critic, real, fake, tensorboard_step):
    writer.add_scalar('Loss Critic', Loss_critic, global_step=tensorboard_step)

    with torch.no_grad():
        fake_imgs_grid = make_grid(fake[:8], normalize=True)
        real_imgs_grid = make_grid(real[:8], normalize=True)
        writer.add_image('Real', real_imgs_grid, global_step=tensorboard_step)
        writer.add_image('Fake', fake_imgs_grid, global_step=tensorboard_step)

def gradient_penalty(critic, real, fake, alpha, train_step, device='cpu'):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)

    interpolated = beta * real + fake.detach() * (1 - beta)
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated, alpha, train_step)

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def save_checkpoint(model, optimizer, filename):
    print(f'-- Saving Checkpoint to => {filename}')
    checkpoint = {
        'state_dict':model.state_dict,
        'optimizer':optimizer.state_dict
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optim, lr):
    print('-- Loading Checkpoint from => {checkpoint_file}')
    checkpoint = torch.load(checkpoint_file)

    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])

    for param_group in optim.param_groups:
        param_group['lr'] = lr


def generate_examples(gen, steps, n=50):
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.rand((1, config.Z_DIM, 1, 1)).to(config.DEVICE)
            img = gen(noise, alpha, steps)
            save_image(img * 0.5 + 0.5, 'saved_examples//step_{steps}_{i}.jpg')
    gen.train()