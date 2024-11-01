import torch
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

# Assuming `image_paths` is already available and contains the image paths.
# Also assuming UTKFace dataset class is already defined.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

image_paths = 'UTKFace'

# Define the train and validation splits
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# Set the device for training (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the dataset using the image paths
dataset = UTKFace(image_paths)

# Calculate the number of samples for training and validation
num_train = round(TRAIN_SPLIT * len(dataset))
num_val = len(dataset) - num_train  # Ensures total = len(dataset)

print('No. of training samples:', num_train)
print('No. of validation samples:', num_val)

# Perform the train-validation split
train_dataset, valid_dataset = random_split(dataset, [num_train, num_val], generator=torch.Generator().manual_seed(42))

# Define batch size
BATCH_SIZE = 32

# Create DataLoader objects for training and validation datasets
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Calculate the number of steps per epoch
train_steps = len(train_dataloader.dataset) // BATCH_SIZE
val_steps = len(val_dataloader.dataset) // BATCH_SIZE

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

# Print the corresponding labels (age, gender, race)
print("Age:", sample_batch["age"][0].item())
print("Gender:", dataset_dict['gender_id'][sample_batch["gender"][0].item()])
print("Race:", dataset_dict['race_id'][sample_batch["race"][0].item()])
