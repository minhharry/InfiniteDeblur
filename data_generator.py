import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import numpy as np
import shutil
from tqdm import tqdm
def move_and_rename_file(source_file, destination_folder, new_filename):
  """Moves a file to a specified destination folder and renames it.

  Args:
    source_file: The path to the source file.
    destination_folder: The path to the destination folder.
    new_filename: The desired new filename.
  """

  destination_file = os.path.join(destination_folder, new_filename)
  shutil.move(source_file, destination_file)


def generate_mountain_array(length, peak_height=1.0, peak_position=0.5):
  """Generates a mountain-like array with elements between 0 and 1.

  Args:
    length: The length of the array.
    peak_height: The height of the peak (between 0 and 1).
    peak_position: The position of the peak (between 0 and 1).

  Returns:
    A NumPy array representing the mountain-like array.
  """

  x = np.linspace(0, 1, length)
  distance_from_peak = np.abs(x - peak_position)
  mountain_array = peak_height - distance_from_peak**2

  # Normalize the array to values between 0 and 1
  mountain_array = mountain_array / np.max(mountain_array)

  return mountain_array




RAW_DATA_PATH = 'ffmpeg_output'
FRAMES_PER_SAMPLE = 7
image_path = glob.glob(os.path.join(RAW_DATA_PATH,'*'))
print("Number of images: ", len(image_path))

transform = transforms.Compose([
    transforms.ToTensor(),
])


PREFIX = f'A/A{FRAMES_PER_SAMPLE}'

if not os.path.exists(os.path.join(f'{PREFIX}' , 'sharp')):
    os.makedirs(os.path.join(f'{PREFIX}' , 'sharp'))
if not os.path.exists(os.path.join(f'{PREFIX}' , 'blur')):
    os.makedirs(os.path.join(f'{PREFIX}' , 'blur'))
if not os.path.exists(os.path.join(f'{PREFIX}' , 'blur_gamma')):
    os.makedirs(os.path.join(f'{PREFIX}' , 'blur_gamma')) 

w = generate_mountain_array(FRAMES_PER_SAMPLE)
sumw = sum(w)

for i in tqdm(range(0, len(image_path), FRAMES_PER_SAMPLE)):
    blur_tensor = torch.zeros(3,1080,1920)
    blur_gamma_tensor = torch.zeros(3,1080,1920)
    for j, img_path in enumerate(image_path[i:i+FRAMES_PER_SAMPLE]):
        img = Image.open(img_path)
        if j == FRAMES_PER_SAMPLE//2:
            img.save(os.path.join(f'{PREFIX}' , 'sharp' , str(i//FRAMES_PER_SAMPLE+1).zfill(6) + '.png'))
        img_tensor = transform(img)
        img_tensor_signal = torch.pow(img_tensor,2.2)
        blur_gamma_tensor += w[j]*img_tensor_signal
        blur_tensor += w[j]*img_tensor

    blur_tensor /= sumw
    blur_gamma_tensor /= sumw
    blur_gamma_tensor =  torch.pow(blur_gamma_tensor,1/2.2)
    blur_img = transforms.ToPILImage()(blur_tensor)
    blur_gamma_img = transforms.ToPILImage()(blur_gamma_tensor)
    blur_img.save(os.path.join(f'{PREFIX}' , 'blur' , str(i//FRAMES_PER_SAMPLE+1).zfill(6) + '.png'))
    blur_gamma_img.save(os.path.join(f'{PREFIX}' , 'blur_gamma' , str(i//FRAMES_PER_SAMPLE+1).zfill(6) + '.png'))