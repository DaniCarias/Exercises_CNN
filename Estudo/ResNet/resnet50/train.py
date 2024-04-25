from data_setup import create_dataloaders_cifa10, create_dog_cat_small_dataloaders, create_dog_cat_full_dataloaders
from resnet50 import ResNetModel
from engine import train_step, validation_step
from torchinfo import summary
import torch
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
EPOCHS = 45

lr=0.05

wandb.init(project="ResNet",
            config={
            "learning_rate": lr,
            "dataset": "CAT_DOG_small",
            "epochs": EPOCHS,
            "loss_fn": "CrossEntropyLoss",
            "optimizer": "SGD",
            "note": "ResNet50"
            })

train_dataloader, validation_dataloader = create_dog_cat_small_dataloaders()


model = ResNetModel(n_layers=[3, 4, 6, 3], img_channels=3, num_classes=2).to(device)
#summary(model, input_size=(4, 3, 224, 224))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


for epoch in range(EPOCHS):
    print(f"---------- Epoch {epoch+1}/{EPOCHS}")
    
    print("\nTraining...")
    train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
    
    print("\nValidation...")
    val_loss, val_acc = validation_step(model, validation_dataloader, loss_fn, device)
    
    print(f"-Summary: Epoch {epoch+1}/{EPOCHS} \n->Train Loss: {train_loss:.4f} \n->Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print("\n\n")

    
    metrics = { "epoch": epoch,
                    "train/loss": train_loss, 
                    "train/accuracy": train_acc,
                    "val/loss": val_loss, 
                    "val/accuracy": val_acc}
    wandb.log(metrics)


""" sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
} """

wandb.finish()