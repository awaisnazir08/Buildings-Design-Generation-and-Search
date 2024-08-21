import numpy as np
import os
import torch
import torch.optim as optim
from torchvision.utils import save_image
from PIL import Image
from ..utils.helper_utils import load_config, load_checkpoint, get_transforms_for_512_image
from ..Generator_Model import Generator

class ImageGenerator:
    def __init__(self, DEVICE = 'cpu'):
        self.device = DEVICE
        self.config = load_config()
        self.generator = Generator(in_channels = self.config['model']['CHANNELS_IMG']).to(self.device)
        self.learning_rate = float(self.config['model']['LEARNING_RATE'])
        
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr =self.learning_rate, betas=(0.5, 0.999))
        
        self.generator.eval()
        
        load_checkpoint(self.config['model']['CHECKPOINT_GEN'], self.generator, self.optimizer_generator, self.learning_rate, self.device)
        
        self.transforms = get_transforms_for_512_image()
    
    def generate_image_from_sketch(self, image):
        
        input_image = np.array(Image.open(image))
        transformed_input = self.transforms(image=input_image)["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            generated_tensor = self.generator(transformed_input)
        
        generated_tensor = generated_tensor * 0.5 + 0.5
        
        return generated_tensor

if __name__ == '__main__':
    image_generator = ImageGenerator()
    print('all good')