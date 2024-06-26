# A file to prepare and download data if needed.
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

IMG_SIZE = 224

# Manual data setup
manual_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


BATCH_SIZE = 16


def create_pizza_steak_sushi_dataloaders():
    
    TRAIN_PATH = "../../data/pizza_steak_sushi/train"
    TEST_PATH = "../../data/pizza_steak_sushi/test"
    
    train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=manual_transform)
    validation_dataset = datasets.ImageFolder(root=TEST_PATH, transform=manual_transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    
    print("Data loaded successfully")
    img_batch, label_batch = next(iter(train_dataloader))
    #print(f"Shape of training data: {img_batch.shape}, {label_batch}\n")
    
    print(f"{len(train_dataset)} training images and {len(validation_dataset)} validation images found.")
    return train_dataloader, validation_dataloader

def create_dog_cat_small_dataloaders():
    
    TRAIN_PATH = "../../data/cats_and_dogs_small/train"
    TEST_PATH = "../../data/cats_and_dogs_small/test"
    
    train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=manual_transform)
    validation_dataset = datasets.ImageFolder(root=TEST_PATH, transform=manual_transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    
    print("Data loaded successfully")
    img_batch, label_batch = next(iter(train_dataloader))
    #print(f"Shape of training data: {img_batch.shape}, {label_batch}\n")
    
    print(f"{len(train_dataset)} training images and {len(validation_dataset)} validation images found.")
    return train_dataloader, validation_dataloader