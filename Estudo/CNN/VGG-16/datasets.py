from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import utils

BATCH_SIZE = 16

basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.TrivialAugmentWide(num_magnitude_bins= 15),
    transforms.ToTensor()    
])

def CIFAR10():
    
    train_dataset = datasets.CIFAR10('./data/CIFAR10', train=True, transform=basic_transform, 
                                      target_transform=None, download=True)
    test_dataset = datasets.CIFAR10('./data/CIFAR10', train=False, transform=basic_transform, 
                                     target_transform=None, download=True)

    print(f"Train dataset: {train_dataset} \nTest dataset: {test_dataset} \nTotal data: {len(train_dataset)+len(test_dataset)}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Labels: {train_dataset.class_to_idx}")
    
    return train_dataloader, test_dataloader


def CIFAR100():
    
    train_dataset = datasets.CIFAR10('./data/CIFAR100', train=True, transform=basic_transform, 
                                      target_transform=None, download=True)
    test_dataset = datasets.CIFAR10('./data/CIFAR100', train=False, transform=basic_transform, 
                                     target_transform=None, download=True)

    print(f"Train dataset: {train_dataset} \nTest dataset: {test_dataset} \nTotal data: {len(train_dataset)+len(test_dataset)}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Labels: {train_dataset.class_to_idx}")
    
    return train_dataloader, test_dataloader


def cat_dog():
    
    train_dataset = datasets.ImageFolder('./data/cat_dog/train', transform=transform)
    test_dataset = datasets.ImageFolder('./data/cat_dog/test', transform=transform)
    
    print(f"Train dataset: {train_dataset} \nTest dataset: {test_dataset} \nTotal data: {len(train_dataset)+len(test_dataset)}")
    #utils.plot_img(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Labels: {train_dataset.class_to_idx}")
    
    return train_dataloader, test_dataloader




