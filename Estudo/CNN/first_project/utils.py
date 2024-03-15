import torch
import os
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import random
import matplotlib.pyplot as plt


def remove_corrupted_images(dir_path):
    for subdir, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                img = Image.open(file_path) # open the image file
                img.verify() # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                print('Bad file:', file_path) # print out the names of corrupt files
                os.remove(file_path)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def create_graph_tensorboard(train_loss, test_loss, train_acc, test_acc, epoch):
    writer = SummaryWriter("runs/1conv_10units_3epochs")
        
    writer.add_scalars(f'loss/', {
        'train': train_loss,
        'test': test_loss,
    }, global_step=epoch)
    
    writer.add_scalars(f'accuracy/', {
        'train': train_acc,
        'test': test_acc,
    }, global_step=epoch)
    
    writer.close()
    
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn, 
               device: torch.device):
    
    loss, acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            # Make predictions with the model
            y_pred = model(X)

            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {model.__class__.__name__, loss.item(), acc}

def plot_img(tensor:torch.Tensor):
    for i in range(10):
        plt.imshow(tensor[random.randint(0, len(tensor)-1)][0].permute(1, 2, 0))
        plt.show()




