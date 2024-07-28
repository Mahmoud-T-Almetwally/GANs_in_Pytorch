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