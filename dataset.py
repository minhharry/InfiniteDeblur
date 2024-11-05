import glob
import os
import random

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class AnimeDataset(Dataset):
    def __init__(self, root_dir='E:\\Downloads\\ImageDatasets', size=256):
        self.root_dir = root_dir
        self.size = size
        self.images_paths = []

        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.aug = v2.Compose([
            v2.Resize(720),
            v2.RandomCrop(size),
            v2.RandomVerticalFlip(),
            v2.RandomHorizontalFlip(),
        ])
        self.aug_blur = [
            v2.GaussianBlur(3, sigma=(0.5, 2.0)),
            v2.GaussianBlur(5, sigma=(0.5, 2.0)),
            v2.GaussianBlur(7, sigma=(0.5, 2.0)),
            v2.Compose([
                v2.Resize(size // 2),
                v2.Resize(size, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            ]),
            v2.Compose([
                v2.Resize(size // 3),
                v2.Resize(size, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            ]),
            v2.Compose([
                v2.Resize(size // 4),
                v2.Resize(size, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            ]),
        ]
        
        for dir in os.listdir(root_dir):
            for file in os.listdir(os.path.join(root_dir, dir)):
                self.images_paths.append(os.path.join(root_dir, dir, file))

    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.images_paths[idx]).convert('RGB')
        img = self.to_tensor(img)
        img = self.aug(img)
        blur = self.aug_blur[random.randint(0, len(self.aug_blur) - 1)](img)
        for _ in range(random.randint(0, 3)):
            blur = self.aug_blur[random.randint(0, len(self.aug_blur) - 1)](blur)
        blur += torch.randn(3, self.size, self.size) * random.uniform(0.01, 0.1)
        blur = torch.clamp(blur, 0, 1)
        return blur, img

class GoProDataset(Dataset):
    def __init__(self, root_dir='E:\\Downloads\\GOPRO_Large\\train', size=256, addnoise=True, mode='train'):
        self.mode = mode
        self.root_dir = root_dir
        self.images = {
            'blur': [],
            'blur_gamma': [],
            'sharp': []
        }
        self.size = size
        self.addnoise = addnoise
        for subdir in os.listdir(root_dir):
            blur_path = os.path.join(root_dir, subdir, 'blur')
            blur_gamma_path = os.path.join(root_dir, subdir, 'blur_gamma')
            sharp_path = os.path.join(root_dir, subdir, 'sharp')
            for (x, y, z) in zip(os.listdir(blur_path), os.listdir(blur_gamma_path), os.listdir(sharp_path)):
                self.images['blur'].append(os.path.join(blur_path, x))
                self.images['blur_gamma'].append(os.path.join(blur_gamma_path, y))
                self.images['sharp'].append(os.path.join(sharp_path, z))
                
        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.aug = v2.Compose([
            v2.RandomVerticalFlip(),
            v2.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.images['blur'])

    def __getitem__(self, idx):
        if self.mode == 'test':
            blur_gamma_image = Image.open(self.images['blur_gamma'][idx])
            sharp_image = Image.open(self.images['sharp'][idx])
            blur_gamma_image = blur_gamma_image.convert('RGB')
            sharp_image = sharp_image.convert('RGB')
            blur_gamma_image = self.to_tensor(blur_gamma_image)
            sharp_image = self.to_tensor(sharp_image)
            return blur_gamma_image, sharp_image
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
        
        blur_gamma_image = self.to_tensor(blur_gamma_image)
        sharp_image = self.to_tensor(sharp_image)

        blur_gamma_image, sharp_image = self.aug(torch.cat([blur_gamma_image.unsqueeze(0), sharp_image.unsqueeze(0)], dim=0))
        
        if self.addnoise:
            noise = torch.randn(3, self.size, self.size) * 0.02
            blur_gamma_image += noise
            
        return blur_gamma_image, sharp_image

class GoProDatasetMulipleLabel(Dataset):
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
    
    dataset = GoProDataset(root_dir='E:\\Downloads\\GOPRO_Large\\test', mode='test')
    
    sample_blur, sample_sharp = dataset[np.random.randint(0, len(dataset))]
    
    fig, axes = plt.subplots(1, 2, figsize=(30, 30), dpi=300)

    axes[0].imshow(sample_blur.squeeze().numpy().transpose(1, 2, 0))
    axes[1].imshow(sample_sharp.squeeze().numpy().transpose(1, 2, 0))
    plt.show()
