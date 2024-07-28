import torch
import numpy as np
from PIL import Image
import config
from torch.utils.data import Dataset, DataLoader
import os


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.classes = os.listdir(self.root_dir)

        for index, name in enumerate(self.classes):
            files = os.listdir(os.path.join(self.root_dir, name))
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.classes[label])

        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
        image = config.both_transform(image=image)['image']
        high_res = config.highres_transform(image=image)['image']
        low_res = config.lowres_transform(image=image)['image']

        return low_res, high_res
    
def test():
    dataset = MyImageFolder(config.DATASET)
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == '__main__':
    test()
    