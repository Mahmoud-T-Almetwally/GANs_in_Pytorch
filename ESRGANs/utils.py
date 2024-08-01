import torch
import os
import config
from PIL import Image
import numpy as np
from torchvision.utils import save_image

def save_checkpoint(model, optimizer, filename):
    print(f'-- Saving Checkpoint to => {filename}')
    checkpoint = {
        'state_dict':model.state_dict(),
        'optimizer':optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optim, lr):
    print('-- Loading Checkpoint from => {checkpoint_file}')
    checkpoint = torch.load(checkpoint_file)

    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])

    for param_group in optim.param_groups:
        param_group['lr'] = lr


def plot_examples(low_res_dir, gen, n):
    files = os.listdir(low_res_dir)

    gen.eval()
    for file in files[:n]:
        image = np.array(Image.open(file))
        image = config.lowres_transform(image=image)['image'].unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            upscaled_img = gen(image)
            save_image(upscaled_img * 0.5 + 0.5, f'saved_examples/{file}')
    gen.train()

def gradient_penalty(critic, real, fake, device='cpu'):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)

    interpolated = beta * real + fake.detach() * (1 - beta)
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)

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