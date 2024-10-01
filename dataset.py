import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class GoProDataset(Dataset):
    def __init__(self, root_dir='E:\\Downloads\\GOPRO_Large\\train', transform=None, size=512):
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
        transalation_step = random.randint(3, 10)
        blur_gamma_image = Image.open(self.images['blur_gamma'][idx])
        sharp_image = Image.open(self.images['sharp'][idx])
        sharp_images = []

        assert blur_gamma_image.size[0] >= self.size + transalation_step and blur_gamma_image.size[1] >= self.size + transalation_step, f"Check image size ({self.images['blur_gamma'][idx]}) in dataset."
        assert sharp_image.size[0] >= self.size + transalation_step and sharp_image.size[1] >= self.size + transalation_step, f"Check image size ({self.images['sharp'][idx]}) in dataset."

        blur_gamma_image = blur_gamma_image.convert('RGB')
        sharp_image = sharp_image.convert('RGB')
        x, y = random.randint(0, blur_gamma_image.size[0]-self.size-transalation_step), random.randint(0, blur_gamma_image.size[1]-self.size-transalation_step)
        blur_gamma_image = blur_gamma_image.crop((x,y,x+self.size,y+self.size))
        sharp_image_src = sharp_image.crop((x,y,x+self.size,y+self.size))
        print(type(blur_gamma_image))
        sharp_image_translate_x_fwd = sharp_image.crop((x+transalation_step,y,x+transalation_step+self.size,y+self.size))
        sharp_image_translate_x_bwd = sharp_image.crop((x-transalation_step,y,x-transalation_step+self.size,y+self.size))
        sharp_image_translate_y_fwd = sharp_image.crop((x,y+transalation_step,x+self.size,y+transalation_step+self.size))
        sharp_image_translate_y_bwd = sharp_image.crop((x,y-transalation_step,x+self.size,y-transalation_step+self.size))
        sharp_image_translate_x_fwd_y_fwd = sharp_image.crop((x+transalation_step,y+transalation_step,x+transalation_step+self.size,y+transalation_step+self.size))
        sharp_image_translate_x_fwd_y_bwd = sharp_image.crop((x+transalation_step,y-transalation_step,x+transalation_step+self.size,y-transalation_step+self.size))
        sharp_image_translate_x_bwd_y_fwd = sharp_image.crop((x-transalation_step,y+transalation_step,x-transalation_step+self.size,y+transalation_step+self.size))
        sharp_image_translate_x_bwd_y_bwd = sharp_image.crop((x-transalation_step,y-transalation_step,x-transalation_step+self.size,y-transalation_step+self.size))
        sharp_images = [sharp_image_src,sharp_image_translate_x_fwd,sharp_image_translate_x_bwd,sharp_image_translate_y_fwd,sharp_image_translate_y_bwd,sharp_image_translate_x_fwd_y_fwd,sharp_image_translate_x_fwd_y_bwd,sharp_image_translate_x_bwd_y_fwd,sharp_image_translate_x_bwd_y_bwd]

        # Transform image and label
        if self.transform:
            blur_gamma_image = self.transform(blur_gamma_image)
            temp = []
            for sharp_sample in sharp_images:
                temp.append(self.transform(sharp_sample))
            sharp_images = temp
            
        return blur_gamma_image, sharp_images

if __name__ == '__main__':   
    import matplotlib.pyplot as plt
    import numpy as np
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    dataset = GoProDataset(transform=transform)
    
    sample_blur, sample_sharps = dataset[np.random.randint(0, len(dataset))]
    print(type(sample_sharps[0]))
    sample_blur = sample_blur.numpy().transpose(1, 2, 0)
    sample = [sample_blur]
    for item in sample_sharps:
        sample.append(item.numpy().transpose(1,2,0))

    fig, axes = plt.subplots(5, 2, figsize=(30, 30))
    for i in range(5):
        for j in range(2):
            axes[i,j].imshow(sample[i+j])
    plt.show()
