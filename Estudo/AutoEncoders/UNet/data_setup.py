import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 16


import torch
from torch.utils.data import Dataset
from PIL import Image
    
class SegmentationDataset(Dataset):
    
    def __init__(self, x_folder, y_folder, transform=None):
        self.x_folder = x_folder
        self.y_folder = y_folder
        self.transform = transform

        self.x_filenames = sorted(os.listdir(x_folder))
        self.y_filenames = sorted(os.listdir(y_folder))
        #print(f"X: {self.x_filenames[:5]}, Y: {self.y_filenames[:5] }")

    def __len__(self):
        return min(len(self.x_filenames), len(self.y_filenames))

    def __getitem__(self, idx):
        x_path = os.path.join(self.x_folder, self.x_filenames[idx])
        y_path = os.path.join(self.y_folder, self.y_filenames[idx])

        x_image = Image.open(x_path)
        y_image = Image.open(y_path)

        if self.transform:
            x_image = self.transform(x_image)
            y_image = self.transform(y_image)

        return x_image, y_image
    
def view_images(dataset):
    num_samples = 5
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 20))

    for i in range(num_samples):
        x_image, y_image = dataset[i]
        
        # Convert tensors to numpy arrays and transpose dimensions
        x_image = x_image.permute(1, 2, 0).numpy()
        y_image = y_image.permute(1, 2, 0).numpy()

        axes[i, 0].imshow(x_image)
        axes[i, 0].set_title('Input Image')

        axes[i, 1].imshow(y_image, cmap='gray')
        axes[i, 1].set_title('Mask')

    plt.tight_layout()
    plt.show()


def create_forest_dataloaders():
    
    transform_Unet = transforms.Compose([
    transforms.Resize((560, 560)),
    transforms.ToTensor()
    ])

    # Define paths to X (images) and Y (masks) folders
    train_x_folder = "../../data/forest_seg/half_size/train/imgs"
    train_y_folder = "../../data/forest_seg/half_size/train/masks"
    vali_x_folder = "../../data/forest_seg/half_size/validation/imgs"
    vali_y_folder = "../../data/forest_seg/half_size/validation/masks"

    # Create dataset
    train_dataset = SegmentationDataset(train_x_folder, train_y_folder, transform=transform_Unet)
    validation_dataset = SegmentationDataset(vali_x_folder, vali_y_folder, transform=transform_Unet)

    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    #print(f"Shape of dataloader: {next(iter(train_dataloader))[0].shape}")
    validation_dataloader = DataLoader(validation_dataset, batch_size=2, shuffle=True)
    #print(f"Shape of dataloader: {next(iter(train_dataloader))[0].shape}")
    
    #view_images(train_dataset)
    
    return train_dataloader, validation_dataloader