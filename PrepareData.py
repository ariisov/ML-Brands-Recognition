import os
import pandas as pd
from PIL import Image
import torch
import clip
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn import functional as F
import albumentations as A
import numpy as np
from os import environ
from dotenv import loadenv
loadenv()

csv_file = environ['CSV_FILE']
data_path = environ['BRANDS_IMAGES']
augmented_data_path = environ['AUG_BRANDS_IMAGES']

# Creating a folder for augmented images, if there is none
if not os.path.exists(augmented_data_path):
    os.makedirs(augmented_data_path)


# Augmentation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomScale(scale_limit=0.1, p=0.2)
])

def find_image_file(filename, folder_path):
    jpg_path = os.path.join(folder_path, filename + ".jpg")
    png_path = os.path.join(folder_path, filename + ".png")
    
    if os.path.exists(jpg_path):
        return jpg_path
    elif os.path.exists(png_path):
        return png_path
    else:
        return None

def augment_image(image_path, brand_name, output_dir, num_augmentations=5):
    image = Image.open(image_path)
    image_np = np.array(image)
    
    augmented_image_paths = []
    for i in range(num_augmentations - 1):  # -1 because the original image is also considered
        augmented = transform(image=image_np)
        augmented_image_np = augmented['image']
        augmented_image = Image.fromarray(augmented_image_np)
        
        # Creating a unique name for the augmented image
        base_name = os.path.basename(image_path)
        file_name, ext = os.path.splitext(base_name)
        augmented_file_name = f"{file_name}_aug_{i+1}{ext}"
        augmented_image_path = os.path.join(output_dir, augmented_file_name)
        
        augmented_image.save(augmented_image_path)
        augmented_image_paths.append(augmented_image_path)
    
    return augmented_image_paths


# Read CSV
df = pd.read_csv(csv_file, sep=';', on_bad_lines='skip', encoding='utf-8')
df['filename'] = df['filename'].astype(str)
new_data = []

# Processing of each brand
for _, row in df.iterrows():
    brand = row['brand']
    filename = row['filename']
    original_image_path = find_image_file(filename, data_path)
    
    if original_image_path and os.path.exists(original_image_path):
        augmented_image_paths = augment_image(original_image_path, brand, augmented_data_path)
        
        new_data.append({'brand': brand, 'image_path': original_image_path})
        
        # Adding augmented images
        for img_path in augmented_image_paths:
            new_data.append({'brand': brand, 'image_path': img_path})

# Saving
augmented_df = pd.DataFrame(new_data)
augmented_df.to_csv(environ['AUG_CSV_FILE'], sep=';', index=False, encoding='utf-8')

print("Аугментация завершена и новый CSV сохранен!")