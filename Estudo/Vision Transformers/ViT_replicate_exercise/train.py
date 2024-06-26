import torch
from vit import VisionTransformer 
from data_setup import create_dog_cat_full_dataloaders
from torchinfo import summary
from engine import train_step, validation_step
import wandb


device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100

lr=0.001


wandb.init(project="ViT",
            config={
            "learning_rate": lr,
            "dataset": "CAT_DOG_full",
            "epochs": EPOCHS,
            "loss_fn": "CrossEntropyLoss",
            "optimizer": "Adam",
            "note": "No Pre-training"
            })


train_dataloader, validation_dataloasder = create_dog_cat_full_dataloaders()


rand_tensor = torch.rand((1, 3, 224, 224)).to(device)

model = VisionTransformer(n_layers=12, d_model=768, heads=12, patch_size=16, in_channels=3, n_classes=2, device=device).to(device)   
#summary(model, input_size=(1, 3, 224, 224))
x = model(rand_tensor)

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
    
    metrics = { "epoch": epoch,
                    "train/loss": train_loss, 
                    "train/accuracy": train_acc,
                    "val/loss": val_loss, 
                    "val/accuracy": val_acc}
    wandb.log(metrics)

wandb.finish()