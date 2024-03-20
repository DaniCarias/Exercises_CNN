import torch
from torch import nn
from torch import utils


def val_fn(model: nn.Module, data_loader: utils.data.DataLoader, loss_fn: nn, accuracy_fn, device: torch.device):
    
    val_loss, val_accuracy = 0, 0
    
    model.eval()
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            
            val_loss += loss_fn(y_pred, y)
            val_accuracy += accuracy_fn(y, y_pred.argmax(dim=1))
            
            
            if batch % 200 == 0:
                print(f"Batch: {batch}/{len(data_loader)}")
        
        
        val_loss /= len(data_loader)
        val_accuracy /= len(data_loader)
        print(f"Validation Loss: {val_loss} | Validation Accuracy: {val_accuracy}")
    
        return val_loss, val_accuracy 