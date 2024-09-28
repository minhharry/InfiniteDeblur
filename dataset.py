import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class GoProDataset(Dataset):
    def __init__(self, root_dir='E:\\Downloads\\GOPRO_Large\\train', transform=None, size=256):
        self.root_dir = root_dir
        self.transform = transform
        self.images = {
            'blur': [],
            'blur_gamma': [],
            'sharp': []
        }
        self.size = size
        for subdir in os.listdir(root_dir):
            blur_path = os.path.join(root_dir, subdir, 'blur')
            blur_gamma_path = os.path.join(root_dir, subdir, 'blur_gamma')
            sharp_path = os.path.join(root_dir, subdir, 'sharp')
            for (x, y, z) in zip(os.listdir(blur_path), os.listdir(blur_gamma_path), os.listdir(sharp_path)):
                self.images['blur'].append(os.path.join(blur_path, x))
                self.images['blur_gamma'].append(os.path.join(blur_gamma_path, y))
                self.images['sharp'].append(os.path.join(sharp_path, z))

    def __len__(self):
        return len(self.images['blur'])

    def __getitem__(self, idx):
        # Load image and label
        blur_gamma_image = Image.open(self.images['blur_gamma'][idx])
        sharp_image = Image.open(self.images['sharp'][idx])

        assert blur_gamma_image.size[0] >= self.size and blur_gamma_image.size[1] >= self.size, f"Check image size ({self.images['blur_gamma'][idx]}) in dataset."
        assert sharp_image.size[0] >= self.size and sharp_image.size[1] >= self.size, f"Check image size ({self.images['sharp'][idx]}) in dataset."

        blur_gamma_image = blur_gamma_image.convert('RGB')
        sharp_image = sharp_image.convert('RGB')
        x, y = random.randint(0, blur_gamma_image.size[0]-self.size), random.randint(0, blur_gamma_image.size[1]-self.size)
        blur_gamma_image = blur_gamma_image.crop((x,y,x+self.size,y+self.size))
        sharp_image = sharp_image.crop((x,y,x+self.size,y+self.size))

        # Transform image and label
        if self.transform:
            blur_gamma_image = self.transform(blur_gamma_image)
            sharp_image = self.transform(sharp_image)
        
        return blur_gamma_image, sharp_image

if __name__ == '__main__':   
    import matplotlib.pyplot as plt
    import numpy as np
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    dataset = GoProDataset(transform=transform)
    
    sample_blur, sample_sharp = dataset[np.random.randint(0, len(dataset))]
    sample_blur = sample_blur.numpy().transpose(1, 2, 0)
    sample_sharp = sample_sharp.numpy().transpose(1, 2, 0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sample_blur)
    axes[1].imshow(sample_sharp)    
    plt.show()