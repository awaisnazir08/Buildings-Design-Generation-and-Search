import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from .discriminator import Discriminator
from .generator import Generator
from .config import LEARNING_RATE, DEVICE, CHECKPOINT_DISC, CHECKPOINT_GEN, CHANNELS_IMG, transform
import torch.optim as optim
from .utils import load_checkpoint
from torchvision.utils import save_image

generator = Generator(in_channels=CHANNELS_IMG).to(DEVICE)

optimizer_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

generator.eval()

load_checkpoint(CHECKPOINT_GEN, generator, optimizer_gen, lr=LEARNING_RATE)

input_dir = 'Pix2Pix_Buildings/sketches_images'
output_dir = 'Pix2Pix_Buildings/generated_images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

images = os.listdir(input_dir)

for image in images:
    image_path = os.path.join(input_dir, image)
    input_image = np.array(Image.open(image_path))

    transformed_input = transform(image=input_image)["image"].unsqueeze(0).to(DEVICE)

    # Perform inference with the generator
    with torch.no_grad():
        generated_tensor = generator(transformed_input)

    generated_tensor = generated_tensor * 0.5 + 0.5 

    # Convert to a numpy array and display it
    # generated_image = generated_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    output_image_name = image[:-10] + 'generated.jpg'
    save_image(generated_tensor, output_dir + '/' + output_image_name)
    print(f'{output_image_name} saved successfully')


