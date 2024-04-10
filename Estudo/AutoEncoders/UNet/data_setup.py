import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 16

transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 224)),
    #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

def create_train_val_datasets():
    IMG_PATH = "../../data/forest_seg/images"
    MASK_PATH = "../../data/forest_seg/masks"
    TRAIN_PATH = "../../data/forest_seg/train"
    VAL_PATH = "../../data/forest_seg/validation"
    
    




    
"""
def create_car_seg_dataloaders():
    
    
    
        
        
    train_dataset = datasets.ImageFolder(root=, transform=transform)
    validation_dataset = datasets.ImageFolder(root=, transform=transform)
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    
    print("Data loaded successfully")
    #img_batch, label_batch = next(iter(train_dataloader))
    #print(f"Shape of training data: {img_batch.shape}, {label_batch}\n")
    
    
    #plot the 4 images

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    for i in range(4):
        img, label = train_dataset[i]
        ax[i].imshow(img.permute(1, 2, 0))
        ax[i].set_title(f"Label: {label}")
        ax[i].axis("off")
    plt.show()
    
    
    
    print(f"{len(train_dataset)} training images and {len(validation_dataset)} validation images found.")
    return train_dataloader, validation_dataloader
"""
    
 


if __name__ == "__main__":
    create_train_val_datasets()

    
    