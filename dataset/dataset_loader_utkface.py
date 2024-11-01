from dataset_loader import UTKFace
import numpy as np
import glob
import os
import torch
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from PIL import ImageFile

# In case your images are truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define the UTKFace folder path
UTKFACE_DIR = "UTKFace"  # Ensure this points to the correct folder

# Fetch all image paths from the UTKFace directory
image_paths = sorted(glob.glob(os.path.join(UTKFACE_DIR, "*.jpg.chip.jpg")))

print(f"Total images found: {len(image_paths)}")

# Define split ratios for train, validation, and test
TRAIN_SPLIT = 0.8  # 80% for training
VALID_SPLIT = 0.1  # 10% for validation
TEST_SPLIT = 0.1   # 10% for testing

# Set the device for training (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming `UTKFace` dataset class is defined and available
# Create the dataset using the image paths
dataset = UTKFace(image_paths)

# Calculate the number of samples for each split
num_train = int(TRAIN_SPLIT * len(dataset))
num_valid = int(VALID_SPLIT * len(dataset))
num_test = len(dataset) - num_train - num_valid  # Remaining for test set

print(f'Train samples: {num_train}')
print(f'Validation samples: {num_valid}')
print(f'Test samples: {num_test}')

# Perform the train-validation-test split
train_dataset, valid_dataset, test_dataset = random_split(
    dataset, [num_train, num_valid, num_test], generator=torch.Generator().manual_seed(42)
)

# Define batch size
BATCH_SIZE = 32

# Create DataLoader objects for each split
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Calculate the number of steps per epoch
train_steps = len(train_dataloader.dataset) // BATCH_SIZE
valid_steps = len(valid_dataloader.dataset) // BATCH_SIZE
test_steps = len(test_dataloader.dataset) // BATCH_SIZE




def imshow(img):
    """
    Helper function to display a tensor image after unnormalizing it.
    """
    img = img / 2 + 0.5  # Unnormalize (reverse the normalization)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Convert from CHW to HWC format
    plt.show()

# Define the dataset_dict to map gender and race IDs to labels
dataset_dict = {
    'gender_id': {
        0: 'Male',
        1: 'Female'
    },
    'race_id': {
        0: 'White',
        1: 'Black',
        2: 'Asian',
        3: 'Indian',
        4: 'Others'
    }
}

# Get a sample batch from the train dataloader
sample_batch = next(iter(train_dataloader))

# Show the first image in the batch
imshow(sample_batch['image'][0])
plt.show()

# Print the corresponding labels (age, gender, race)
print("Age:", sample_batch["age"][0].item())
print("Gender:", dataset_dict['gender_id'][sample_batch["gender"][0].item()])
print("Race:", dataset_dict['race_id'][sample_batch["race"][0].item()])

# Get a sample batch from the train dataloader
sample_batch = next(iter(train_dataloader))

# Show the first image in the batch and print its labels
imshow(sample_batch['image'][0])
plt.show()

# Print the corresponding labels (age, gender, race)
print("Age:", sample_batch["age"][0].item())
print("Gender:", dataset_dict['gender_id'][sample_batch["gender"][0].item()])
print("Race:", dataset_dict['race_id'][sample_batch["race"][0].item()])
