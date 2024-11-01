from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
import os

# In case your images are truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

class UTKFace(Dataset):
    def __init__(self, image_paths):
        # Define the mean and standard deviation for ImageNet pre-trained models
        mean = [0.485, 0.456, 0.406]  # ImageNet mean
        std = [0.229, 0.224, 0.225]   # ImageNet std

        # Define the transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224 (standard for models like ResNet)
            transforms.ToTensor(),          # Convert image to PyTorch tensor
            transforms.Normalize(mean=mean, std=std)  # Normalize with ImageNet statistics
        ])

        # Set Inputs and Labels
        self.image_paths = image_paths
        self.images = []
        self.ages = []
        self.genders = []
        self.races = []

        # Parse file names to extract labels (age, gender, race)
        for path in image_paths:
            filename = os.path.basename(path).split("_")  # Cross-platform path handling
            try:
                # Check if the filename has the expected format
                if len(filename) == 4:
                    self.images.append(path)
                    self.ages.append(int(filename[0]))     # Age
                    self.genders.append(int(filename[1]))  # Gender
                    self.races.append(int(filename[2]))    # Race
                else:
                    print(f"Skipping file {path}: Incorrect filename format")
            except ValueError:
                print(f"Skipping file {path}: Unable to parse labels")
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load an image
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
        
        # Apply the transformations
        img = self.transform(img)

        # Get the corresponding labels
        age = self.ages[index]
        gender = self.genders[index]
        race = self.races[index]
        
        # Prepare the sample dictionary
        sample = {
            'image': img,
            'age': age,
            'gender': gender,
            'race': race
        }

        return sample
