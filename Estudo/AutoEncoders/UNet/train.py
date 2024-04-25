import torch
import wandb
from data_setup import create_forest_dataloaders
from UNet import UNet
from engine import train_step, validation_step

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

EPOCHS = 15
LR = 0.01

wandb.init(project="UNet",
            config={
            "learning_rate": LR,
            "dataset": "forest",
            "epochs": EPOCHS,
            "loss_fn": "CrossEntropyLoss",
            "optimizer": "Adam",
            "note": "no DropOut and BatchNorm"
            })


train_dataloader, validation_dataloader = create_forest_dataloaders()

#model = UNet(1, 2).to(device) 
model = UNet(3, 3).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    print(f"---------- Epoch {epoch+1}/{EPOCHS}")
    
    print("\nTraining...")
    train_loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
    
    print("\nValidation...")
    val_loss = validation_step(model, validation_dataloader, loss_fn, device)
    
    print(f"-Summary: Epoch {epoch+1}/{EPOCHS} \n->Train Loss: {train_loss:.4f}")
    print("\n\n")

    
    metrics = { "epoch": epoch,
                    "train/loss": train_loss, 
                    "val/loss": val_loss}
    wandb.log(metrics)


wandb.finish()
