import torch
import wandb
from data_setup import create_car_seg_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

EPOCHS = 15
LR = 0.01

""" wandb.init(project="UNet",
            config={
            "learning_rate": LR,
            "dataset": "",
            "epochs": EPOCHS,
            "loss_fn": "",
            "optimizer": "",
            "note": "no DropOut and BatchNorm"
            }) """


train_dataloader, validation_dataloader = create_car_seg_dataloaders()
