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

# discriminator = Discriminator(in_channels=CHANNELS_IMG).to(DEVICE)
generator = Generator(in_channels=CHANNELS_IMG).to(DEVICE)
# optimizer_disc = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Load the checkpoints
load_checkpoint(CHECKPOINT_GEN, generator, optimizer_gen, lr=LEARNING_RATE)
# load_checkpoint(CHECKPOINT_DISC, discriminator, optimizer_disc, lr=LEARNING_RATE)

input_dir = 'Pix2Pix_Buildings/sketches_images'

images = os.listdir(input_dir)
input_image = np.array(Image.open(input_image_path))

transformed_input = transform(image=input_image)["image"].unsqueeze(0).to(DEVICE)

# Perform inference with the generator
generator.eval()
with torch.no_grad():
    generated_tensor = generator(transformed_input)

generated_tensor = generated_tensor * 0.5 + 0.5 

# Convert to a numpy array and display it
generated_image = generated_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
plt.imshow(generated_image)
plt.axis('off')
plt.show()


