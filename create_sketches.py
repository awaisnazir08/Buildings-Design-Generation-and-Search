# import cv2
# import matplotlib.pyplot as plt
# import os

# input_dir = 'vyronas-database\Vyronasdbmin'
# input_images = os.listdir(input_dir)
# output_image_folder = 'sketches_images'
# for image in input_images:
#     image_path = os.path.join(input_dir, image)
#     im  = cv2.imread(image_path)
#     gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     inverted_gray_img = cv2.bitwise_not(gray_img)
#     blurred = cv2.GaussianBlur(inverted_gray_img, (21, 21), 0)
#     Invertedblur = cv2.bitwise_not(blurred)
#     sketch = cv2.divide(gray_img, Invertedblur, scale=256.0)
#     sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
#     file_name = output_image_folder + '/' + image[:-4] + '_sketch' + '.jpg'
#     cv2.imwrite(file_name,  sketch_rgb)
# print(sketch_rgb.shape)
# plt.imshow(sketch_rgb)
# plt.axis('off')
# plt.show()


import torch
from torchvision import transforms
from PIL import Image

# Load your image (assuming it's a PIL Image)
input_image = Image.open("Pix2Pix_buildings/Test Pictures/building12_low_noon.jpg")

# Define the transformation to resize the image
resize_transform = transforms.Resize((512, 512))

# Apply the transformation
resized_image = resize_transform(input_image)

# Convert to tensor if needed for further processing
image_tensor = transforms.ToTensor()(resized_image)
resized_image.show()  # This will open the resized RGB image in the default image viewer


