import torch
from torch import nn
from data_setup import create_dog_cat_small_dataloaders
from torchinfo import summary
from engine import train_step, validation_step
import wandb
import torchvision


device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 16

lr=0.05

"""
wandb.init(project="ResNet pytorch",
            config={
            "learning_rate": lr,
            "dataset": "CAT_DOG_small",
            "epochs": EPOCHS,
            "loss_fn": "CrossEntropyLoss",
            "optimizer": "SGD",
            "note": "ResNet50 torchvision With Pre-training"
            })
"""

train_dataloader, validation_dataloasder = create_dog_cat_small_dataloaders()



weights = torchvision.models.ResNet50_Weights.DEFAULT
model = torchvision.models.resnet50(weights=weights).to(device)

for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(in_features=2048, out_features=2, device=device)

#summary(model, input_size=(1, 3, 224, 224)) # To check if the model architecture
print(model)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


for epoch in range(EPOCHS):
    print(f"---------- Epoch {epoch+1}/{EPOCHS}")
    
    print("\nTraining...")
    train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
    
    print("\nValidation...")
    val_loss, val_acc = validation_step(model, validation_dataloasder, loss_fn, device)
    
    print(f"\n-Summary: Epoch {epoch+1}/{EPOCHS} \n->Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} \n->Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print("\n\n")
    """
    metrics = { "epoch": epoch,
                "train/loss": train_loss, 
                "train/accuracy": train_acc,
                "val/loss": val_loss, 
                "val/accuracy": val_acc}
    wandb.log(metrics)

wandb.finish()"""