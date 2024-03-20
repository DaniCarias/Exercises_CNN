# A file to prepare and download data if needed.
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision
import torch

# Manual data setup
manual_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# Automatic data setup
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transform = weights.transforms()


BATCH_SIZE = 16


def create_cat_dog_dataloaders():
    
    TRAIN_PATH = "../data/cats_dogs_full/train"
    TEST_PATH = "../data/cats_dogs_full/test"
    
    train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=manual_transform)
    validation_dataset = datasets.ImageFolder(root=TEST_PATH, transform=manual_transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return train_dataloader, validation_dataloader


def create_pizza_steak_sushi_dataloaders():
    
    TRAIN_PATH = "../data/pizza_steak_sushi/train"
    TEST_PATH = "../data/pizza_steak_sushi/test"
    
    train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=auto_transform)
    validation_dataset = datasets.ImageFolder(root=TEST_PATH, transform=auto_transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return train_dataloader, validation_dataloader