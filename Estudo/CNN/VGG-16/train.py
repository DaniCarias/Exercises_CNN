import torch 
from torch import nn
from torch import utils


def train_fn(model: nn.Module, data_loader: utils.data.DataLoader, loss_fn: nn, optimizer: torch.optim, accuracy_fn, device):
    
    train_loss, train_accuracy = 0, 0
    
    model.train()
    
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)
        
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_accuracy += accuracy_fn(y, y_pred.argmax(dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        
        if batch % 500 == 0:
            print(f"Batch: {batch}/{len(data_loader)}")
            
    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)
    print(f"Train Loss: {train_loss} | Train Accuracy: {train_accuracy}")
    
    return train_loss, train_accuracy