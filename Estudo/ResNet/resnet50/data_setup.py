import torchvision
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
import os
import matplotlib.pyplot as plt
import random


BATCH_SIZE = 16
    
transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def create_dataloaders_cifa10():

    train_dataset = datasets.CIFAR10('../../data/CIFAR10', train=True, download=True)
    validation_dataset = datasets.CIFAR10('../../data/CIFAR10', train=False, download=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)
    
    
    print("Data loaded successfully")
    print(f"Classes: {train_dataset.classes}")
    print(f"Shape of train_dataset: {train_dataset.data.shape}")

    
    return train_dataloader, validation_dataloader
    
    
def create_dog_cat_small_dataloaders():
    
    TRAIN_PATH = "../../data/cats_and_dogs_small/train"
    TEST_PATH = "../../data/cats_and_dogs_small/test"
    
    train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
    validation_dataset = datasets.ImageFolder(root=TEST_PATH, transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    
    print("Data loaded successfully")
    #img_batch, label_batch = next(iter(train_dataloader))
    #print(f"Shape of training data: {img_batch.shape}, {label_batch}\n")
    
    print(f"{len(train_dataset)} training images and {len(validation_dataset)} validation images found.")
    return train_dataloader, validation_dataloader

def create_dog_cat_full_dataloaders():
    
    TRAIN_PATH = "../../data/cats_dogs_full/train"
    TEST_PATH = "../../data/cats_dogs_full/test"
    
    train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
    validation_dataset = datasets.ImageFolder(root=TEST_PATH, transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    
    print("Data loaded successfully")
    #img_batch, label_batch = next(iter(train_dataloader))
    #print(f"Shape of training data: {img_batch.shape}, {label_batch}\n")
    
    print(f"{len(train_dataset)} training images and {len(validation_dataset)} validation images found.")
    return train_dataloader, validation_dataloader


