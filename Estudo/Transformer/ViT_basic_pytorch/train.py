import torch
from torch import nn
from vit import ViT
from data_setup import create_dog_cat_small_dataloaders
from torchinfo import summary
from engine import train_step, validation_step
import torchvision
import wandb


device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 6

lr=1e-3


wandb.init(project="ViT",
            config={
            "learning_rate": lr,
            "dataset": "CAT_DOG_small",
            "epochs": EPOCHS,
            "loss_fn": "CrossEntropyLoss",
            "optimizer": "Adam",
            "note": "Pre-training"
            })


train_dataloader, validation_dataloasder = create_dog_cat_small_dataloaders()






weights = torchvision.models.ViT_B_16_Weights.DEFAULT
model_pretreined = torchvision.models.vit_b_16(weights=weights).to(device)

for params in model_pretreined.parameters():
    params.requires_grad = False
    
model_pretreined.heads = nn.Sequential(
    nn.LayerNorm(normalized_shape=768),
    nn.Linear(in_features=768, out_features=3)
).to(device)

#summary(vit_pretrained, input_size=(1, 3, 224, 224))





""" model = ViT(img_size=224, in_channels=3, patch_size=16, embedding_dim=768, mlp_size=3072,
            heads=12, num_encoder_layers=12, device=device, dropout=0.1, num_classes=3).to(device)
 """
#summary(model, input_size=(1, 3, 224, 224)) # To check if the model architecture and the parameters are correct

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_pretreined.parameters(), lr=lr)




for epoch in range(EPOCHS):
    print(f"---------- Epoch {epoch+1}/{EPOCHS}")
    
    print("\nTraining...")
    train_loss, train_acc = train_step(model_pretreined, train_dataloader, loss_fn, optimizer, device)
    
    print("\nValidation...")
    val_loss, val_acc = validation_step(model_pretreined, validation_dataloasder, loss_fn, device)
    
    print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print("\n\n")
    
    metrics = { "epoch": epoch,
                    "train/loss": train_loss, 
                    "train/accuracy": train_acc,
                    "val/loss": val_loss, 
                    "val/accuracy": val_acc}
    wandb.log(metrics)

wandb.finish()