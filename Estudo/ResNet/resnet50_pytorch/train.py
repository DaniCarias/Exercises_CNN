import torch
from torch import nn
from resnet50 import ResNetModel
from data_setup import create_dog_cat_small_dataloaders
from torchinfo import summary
from engine import train_step, validation_step
import wandb
import torchvision


device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 6

lr=1e-3


""" wandb.init(project="ResNet",
            config={
            "learning_rate": lr,
            "dataset": "CAT_DOG_small",
            "epochs": EPOCHS,
            "loss_fn": "CrossEntropyLoss",
            "optimizer": "Adam",
            "note": "ResNet50"
            }) """


train_dataloader, validation_dataloasder = create_dog_cat_small_dataloaders()



model = ResNetModel(n_blocks=[3, 4, 6, 3], n_channels=[64, 128, 256,512], 
                    bootlenecks=[256, 512, 1024, 2048], img_channels=3,
                    first_kernel_size=7, out_features=2).to(device)

#model = torchvision.models.resnet50()

summary(model, input_size=(1, 3, 224, 224)) # To check if the model architecture


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


for epoch in range(EPOCHS):
    print(f"---------- Epoch {epoch+1}/{EPOCHS}")
    
    print("\nTraining...")
    train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
    
    print("\nValidation...")
    val_loss, val_acc = validation_step(model, validation_dataloasder, loss_fn, device)
    
    print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print("\n\n")
    
"""     metrics = { "epoch": epoch,
                    "train/loss": train_loss, 
                    "train/accuracy": train_acc,
                    "val/loss": val_loss, 
                    "val/accuracy": val_acc}
    wandb.log(metrics)

wandb.finish() """