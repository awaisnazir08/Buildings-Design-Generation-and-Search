import torch
from PIL import Image
import numpy as np

class Embeddings: 
    def __init__(self, model, preprocess, device = 'cpu'):
        self.model = model
        self. preprocess = preprocess
        self. device = device
    
    def generate_image_embedding(self, image):
        # Check if the input is a path (string) or a NumPy array
        if isinstance(image, str):
            # Load the image from the file path
            image_pil = Image.open(image)
        elif isinstance(image, np.ndarray):
            # Convert the NumPy array to a PIL Image
            image_pil = Image.fromarray(image)
        else:
            raise ValueError("The input must be a file path or a NumPy array")

        # Preprocess the image using the provided preprocessing function
        image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_embedding = self.model.encode_image(image_tensor)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        return image_embedding.cpu().numpy()[0]
