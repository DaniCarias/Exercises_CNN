from data_process import create_data_loaders, create_FashionMNIST_data_loaders
from utils import accuracy_fn, remove_corrupted_images, create_graph_tensorboard, eval_model
import torch
from torch import nn
import matplotlib.pyplot as plt
import wandb

class ModelCNN(nn.Module):
    def __init__(self, input_shape, units_shape, output_shape):
        super().__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=units_shape, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=units_shape, out_channels=units_shape, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=units_shape, out_channels=units_shape, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=units_shape, out_channels=units_shape, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=units_shape*32*32, out_features=output_shape)
            )
        
    def forward(self, x):
        x = self.conv_block1(x)
        #print(x.shape)
        x = self.conv_block2(x)
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        return x
    
        
def train_model(model, data_loader, loss_fn, optimizer):
    model.train()
    
    train_loss = 0
    train_acc = 0
    
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        
        train_pred = model(X)
        
        #print(f"Train_pred: {train_pred} Logit_pred: {y_logit} | y: {y}")
        #print(f"Train Pred: {train_pred.shape} | y: {y.shape}")
        #print(f"Train Pred: {train_pred.dtype} | y: {y.dtype}")
        
        loss = loss_fn(train_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y, train_pred.argmax(dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if batch % 400 == 0:
            print(f"Batch: {batch}/{len(data_loader)}")
    
    train_acc /= len(data_loader)
    train_loss /= len(data_loader)
    print(f"Per batch -> Train Loss: {train_loss} | Train Accuracy: {train_acc:.2f}%")
    
    return train_loss, train_acc
        
        
def test_model(model, data_loader, loss_fn):
    model.eval()
    
    test_loss = 0
    test_acc = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            
            test_pred = model(X)
            
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y, test_pred.argmax(dim=1))
            
            if batch % 50 == 0:
                print(f"Batch: {batch}/{len(data_loader)}")
        
        test_acc /= len(data_loader)
        test_loss /= len(data_loader)
        
        print(f"Per batch -> Test Loss: {test_loss} | Test Accuracy: {test_acc:.2f}%")
        
        return test_loss, test_acc
        


if __name__ == '__main__':
    train_dir = "./data/cats_dogs_full/train"
    test_dir = "./data/cats_dogs_full/test"
    remove_corrupted_images(train_dir)
    remove_corrupted_images(test_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    epochs = 30
    lr = 0.01
    units_shape = 25
    
    wandb.init(
        project="Cat_dog_full",

        config={
        "learning_rate": lr,
        "num_conv_blocks": 2,
        "dataset": "CatDogFull",
        "epochs": epochs,
        "units_shape": units_shape,
        "TrivialAugmentWide": 15,
        "loss_fn": "CrossEntropyLoss",
        "kernel_size": 3,
        })
    
    model = ModelCNN(input_shape=3, units_shape=units_shape, output_shape=2).to(device)

    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    train_dataloader, test_dataloader = create_data_loaders(train_dir, test_dir)
    #train_dataloader, test_dataloader = create_FashionMNIST_data_loaders()
    
    
    for epoch in range(epochs):
        print(f"\n--------Epoch: {epoch}")
        
        print(f"-Training...")
        train_loss, train_acc = train_model(model, train_dataloader, loss_fn, optimizer)
        
        print(f"-Testing...")
        test_loss, test_acc = test_model(model, test_dataloader, loss_fn)
        
        metrics = { "epoch": epoch,
                    "train/loss": train_loss, 
                    "train/accuracy": train_acc,
                    "test/loss": test_loss, 
                    "test/accuracy": test_acc}
        wandb.log(metrics)
                
    model, loss, accuracy  = eval_model(model, test_dataloader, loss_fn, accuracy_fn, device)
    print(f"------ Final Results ------")
    print(f"Model: {model} | Loss: {loss} | Accuracy: {accuracy}")

    wandb.finish()
    