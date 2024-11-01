import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os

# Dictionary to map gender and race IDs to labels
dataset_dict = {
    'gender_id': {0: 'male', 1: 'female'},
    'race_id': {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'}
}

# Get sorted list of image paths from UTKFace dataset
image_paths = sorted(glob.glob("UTKFace/*.jpg.chip.jpg"))

# Initialize lists for images and their respective labels
images = []
ages = []
genders = []
races = []

# Print the number of images found
print(f"Number of image paths found: {len(image_paths)}")

# Ensure that image_paths is not empty before trying to load images
if len(image_paths) == 0:
    raise FileNotFoundError("No images found in the specified directory. Check the path and file names.")

# Proceed with loading the images
for path in image_paths:
    filename = os.path.basename(path).split("_")
    if len(filename) == 4:
        img = np.array(Image.open(path))  # Attempt to load the image
        if img is not None:
            images.append(img)
            ages.append(int(filename[0]))
            genders.append(int(filename[1]))
            races.append(int(filename[2]))
        else:
            print(f"Failed to load image: {path}")

print(f"Number of images successfully loaded: {len(images)}")


# Randomly select an index to display an image
idx = np.random.randint(len(images))

# Display the image at the selected index
plt.imshow(images[idx])
plt.show()

# Print corresponding age, gender, and race
print(f"Age: {ages[idx]}")
print(f"Gender: {dataset_dict['gender_id'][genders[idx]]}")
print(f"Race: {dataset_dict['race_id'][races[idx]]}")

# Age normalization and log transformation
min_age_value, max_age_value = min(ages), max(ages)
log_age_values = np.log10(ages)
max_age_log_value = log_age_values.max()

# Display age statistics
print('MAX AGE VALUE:', max_age_value)
print('MIN AGE VALUE:', min_age_value)
print('MAX AGE LOG VALUE:', max_age_log_value)

# Functions to calculate normalized and log-transformed age values
def get_normalized_age_value(original_age_value):
    return (original_age_value - min_age_value) / (max_age_value - min_age_value)

def get_log_age_value(original_age_value):
    return np.log10(original_age_value) / max_age_log_value

# Functions to revert the normalized and log-transformed age values back to original
def get_original_age_from_log_value(log_age_value):
    return np.exp(log_age_value) * max_age_log_value

def get_original_age_value(normalized_age_value):
    return normalized_age_value * (max_age_value - min_age_value) + min_age_value

# Example of normalized age for a given index
normalized_age = get_normalized_age_value(ages[idx])
print(f"Normalized Age: {normalized_age}")
