from utils import plot_img
import torch
import os
import random
from shutil import copy2 
import torchvision
from torchvision import transforms
import requests
import zipfile
from pathlib import Path
import random
from torch.utils.data import DataLoader
from torchvision import datasets

BATCH_SIZE = 32
transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.TrivialAugmentWide(num_magnitude_bins= 15),
            transforms.ToTensor()
            ])

transform_basic = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
            ])


def create_train_test_split(source_dir, train_dir, test_dir, split_ratio=0.8):
    """
    Creates train and test folders with balanced cat and dog images from a source directory.

    Args:
        source_dir (str): Path to the directory containing "cats" and "dogs" folders.
        train_dir (str): Path to the directory to create for training data.
        test_dir (str): Path to the directory to create for testing data.
        split_ratio (float, optional): Ratio of data to allocate for training (default 0.8).
    """

    cat_imgs = [os.path.join(source_dir, "Cat", img) for img in os.listdir(os.path.join(source_dir, "Cat"))]
    dog_imgs = [os.path.join(source_dir, "Dog", img) for img in os.listdir(os.path.join(source_dir, "Dog"))]

    # Randomly shuffle images to ensure balanced distribution in splits
    random.shuffle(cat_imgs)
    random.shuffle(dog_imgs)

    # Calculate number of images for train and test sets based on split_ratio
    num_samples = len(cat_imgs) + len(dog_imgs)
    test_size = int(num_samples * (1 - split_ratio))
    train_size = num_samples - test_size

    # Ensure equal number of cat and dog images in both train and test sets
    min_imgs_per_class = min(len(cat_imgs), len(dog_imgs))
    train_cat_imgs = random.sample(cat_imgs, min_imgs_per_class * train_size // (2 * min_imgs_per_class))
    train_dog_imgs = random.sample(dog_imgs, min_imgs_per_class * train_size // (2 * min_imgs_per_class))
    test_cat_imgs = [img for img in cat_imgs if img not in train_cat_imgs]
    test_dog_imgs = [img for img in dog_imgs if img not in train_dog_imgs]

    # Create train and test folders
    train_cat_dir = os.path.join(train_dir, "cat")
    train_dog_dir = os.path.join(train_dir, "dog")
    test_cat_dir = os.path.join(test_dir, "cat")
    test_dog_dir = os.path.join(test_dir, "dog")

    os.makedirs(train_cat_dir, exist_ok=True)
    os.makedirs(train_dog_dir, exist_ok=True)
    os.makedirs(test_cat_dir, exist_ok=True)
    os.makedirs(test_dog_dir, exist_ok=True)

    # Copy images to train and test folders, maintaining directory structure
    for img in train_cat_imgs:
        copy2(img, train_cat_dir)
    for img in train_dog_imgs:
        copy2(img, train_dog_dir)
    for img in test_cat_imgs:
        copy2(img, test_cat_dir)
    for img in test_dog_imgs:
        copy2(img, test_dog_dir)

    print(f"Successfully created train and test folders with balanced data distribution.")


def create_data_loaders(train_dir, test_dir):
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Dataset: {train_dir} and {test_dir}")
    print(f"Train dataloader: {len(train_dataloader)} batches")
    print(f"Test dataloader: {len(test_dataloader)} batches")
    
    #plot_img(train_dataset)
    
    return train_dataloader, test_dataloader


def create_FashionMNIST_data_loaders():
    
    train_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                         train=True,
                                                         transform=transform,
                                                         download=True)
    
    test_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                         train=False,
                                                         transform=transform,
                                                         download=True)
    
    print(f"Image size: {train_dataset[0][0].shape}")
    #plot_img(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Labels: {len(train_dataset.class_to_idx)}")   
    return train_dataloader, test_dataloader


def create_pizza_steak_sushi_data_loaders():
    
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak, sushi data
        with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
            request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
            print("Downloading pizza, steak, sushi data...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
            print("Unzipping pizza, steak, sushi data...")
            zip_ref.extractall(image_path)


    train_dataset = datasets.ImageFolder(root=train_dir, # Diretoria dos dados para train
                                  transform=transform, # Para mudar o size, rotação e mudar para tensor
                                  target_transform=None) # A transform para as labels/classes

    test_dataset = datasets.ImageFolder(root=test_dir, # Diretoria dos dados para train
                                 transform=transform) # Para mudar o size, rotação e mudar para tensor
        
    print(f"Labels: {train_dataset.classes}")    
    
    BATCH_SIZE = 1 # Porque é um dataset muito pequeno

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
        
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    """
    source_dir = "./data/cats_dogs_full"
    train_dir = "./data/cats_dogs_full/train"
    test_dir = "./data/cats_dogs_full/test"

    create_train_test_split(source_dir, train_dir, test_dir)
    """
    