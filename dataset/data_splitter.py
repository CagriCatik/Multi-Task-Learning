import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from torchvision import transforms

# In case your images are truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define the UTKFace dataset class
class UTKFace(Dataset):
    def __init__(self, image_paths):
        # Define the mean and std for normalization (ImageNet values are common)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Define the transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),          # Convert image to PyTorch tensor
            transforms.Normalize(mean=mean, std=std)  # Normalize with ImageNet stats
        ])

        # Initialize lists for images and labels (age, gender, race)
        self.image_paths = image_paths
        self.ages = []
        self.genders = []
        self.races = []

        # Parse the filenames to extract labels (age, gender, race)
        for path in image_paths:
            filename = os.path.basename(path).split("_")
            if len(filename) == 4:
                self.ages.append(int(filename[0]))
                self.genders.append(int(filename[1]))
                self.races.append(int(filename[2]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load the image
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format

        # Apply transformations
        image = self.transform(image)

        # Get the labels
        age = self.ages[index]
        gender = self.genders[index]
        race = self.races[index]

        # Return the image and labels as a dictionary
        return {
            'image': image,
            'age': age,
            'gender': gender,
            'race': race
        }

# Define the UTKFace folder path
UTKFACE_DIR = "UTKFace/"  # Ensure this points to your dataset folder

# Fetch all image paths from the UTKFace directory
image_paths = sorted(glob.glob(os.path.join(UTKFACE_DIR, "*.jpg.chip.jpg")))  # Adjust file extension if needed

print(f"Total images found: {len(image_paths)}")

# Define split ratios for train, validation, and test
TRAIN_SPLIT = 0.8  # 80% for training
VALID_SPLIT = 0.1  # 10% for validation
TEST_SPLIT = 0.1   # 10% for testing

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

# Get a sample batch from the train dataloader
sample_batch = next(iter(train_dataloader))

# Show the first image in the batch and print its labels
imshow(sample_batch['image'][0])
plt.show()

# Define a dictionary to map gender and race IDs to their labels (for visualization purposes)
dataset_dict = {
    'gender_id': {0: 'male', 1: 'female'},
    'race_id': {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'}
}

# Print the corresponding labels (age, gender, race)
print("Age:", sample_batch["age"][0].item())
print("Gender:", dataset_dict['gender_id'][sample_batch["gender"][0].item()])
print("Race:", dataset_dict['race_id'][sample_batch["race"][0].item()])
