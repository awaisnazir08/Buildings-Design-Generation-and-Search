import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import config

class MapDataset(Dataset):
    def __init__(self, real_images_dir, sketches_dir):
        self.real_images_dir = real_images_dir
        self.sketches_dir = sketches_dir
        self.real_images_files = os.listdir(self.real_images_dir)
        self.sketches_files = os.listdir(self.sketches_dir)
        # print(self.list_files)
    
    def __len__(self):
        return len(self.real_images_files)
    
    def __getitem__(self, index):
        real_img_file = self.real_images_files[index]
        real_img_path = os.path.join(self.real_images_dir, real_img_file)
        real_image = np.array(Image.open(real_img_path))
        
        # sketch_image_file = self.sketches_files[index]
        sketch_image_file = real_img_file[:-4] + '_sketch.jpg'
        sketch_img_path = os.path.join(self.sketches_dir, sketch_image_file)
        sketch_image = np.array(Image.open(sketch_img_path))
        
        augmentations = config.both_transform(image= sketch_image, image0= real_image)
        input_image, target_image = augmentations['image'], augmentations['image0']
        
        input_image = config.transform_only_input(image = input_image)['image']
        
        target_image = config.transform_only_mask(image = target_image)['image']
        
        return input_image, target_image