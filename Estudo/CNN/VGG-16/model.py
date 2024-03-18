import torch
from torch import nn
from train import train_fn
from test import test_fn
import utils
import wandb
import datasets 

class VGG_16_Model(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            #nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=output_shape),
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        #print(f"Shape of x: {x.shape}")
        x = self.classifier(x)
        return x
    
    
    
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    epochs = 8
    lr = 0.005
    
    wandb.init(project="VGG-16",
            config={
            "learning_rate": lr,
            "dataset": "CAT_DOG",
            "epochs": epochs,
            "TrivialAugmentWide": 15,
            "loss_fn": "CrossEntropyLoss",
            "note": ""
            })
    
    model = VGG_16_Model(input_shape=3, output_shape=2)
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    print(f"Creating dataloaders...")
    train_dataloader, test_dataloader = datasets.cat_dog()
        
    for epoch in range(epochs):
        print(f"\n---- Epoch: {epoch}/{epochs}")
        
        
        print(f"Start Training...")
        train_loss, train_acc = train_fn(model, train_dataloader, loss_fn, optimizer, utils.accuracy_fn, device)
        
        print(f"Start Testing...")
        test_loss, test_acc = test_fn(model, test_dataloader, loss_fn, utils.accuracy_fn, device)
           
    
        metrics = { "epoch": epoch,
                    "train/loss": train_loss, 
                    "train/accuracy": train_acc,
                    "test/loss": test_loss, 
                    "test/accuracy": test_acc}
        wandb.log(metrics)
    
    model, loss, accuracy  = utils.eval_model(model, test_dataloader, loss_fn, utils.accuracy_fn, device)
    print(f"------ Final Results ------")
    print(f"Model: {model} | Loss: {loss} | Accuracy: {accuracy}")
    
    wandb.finish()
    
if __name__ == "__main__":
    main()