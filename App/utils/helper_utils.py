from PIL import Image
import torch
from albumentations.pytorch import ToTensorV2
import yaml
import albumentations as A
def load_config(config_path = 'App/config.yaml'):
    """Load the configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def process_user_image(image):
    return Image.fromarray(image).convert('RGB')

def load_checkpoint(checkpoint_file, model, optimizer, lr, DEVICE):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def get_transforms_for_512_image():
    transform_512 = A.Compose(
    [
        A.Resize(width = 512, height = 512),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2()])
    
    return transform_512