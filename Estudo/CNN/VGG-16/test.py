import torch
from torch import nn
from torch import utils


def test_fn(model: nn.Module, data_loader: utils.data.DataLoader, loss_fn: nn, accuracy_fn, device: torch.device):
    
    test_loss, test_accuracy = 0, 0
    
    model.eval()
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            
            test_loss += loss_fn(y_pred, y)
            test_accuracy += accuracy_fn(y, y_pred.argmax(dim=1))
            
            
            if batch % 200 == 0:
                print(f"Batch: {batch}/{len(data_loader)}")
        
        
        test_loss /= len(data_loader)
        test_accuracy /= len(data_loader)
        print(f"Test Loss: {test_loss} | Test Accuracy: {test_accuracy}")
    
        return test_loss, test_accuracy 