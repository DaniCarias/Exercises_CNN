# A file containing various training functions.

import torch

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):

  # Put model in train mode
  model.train()
  
  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0
  
  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate  and accumulate loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)
      
      if batch % 100 == 0:
            print(f"Batch {batch}/{len(dataloader)}")

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  
  print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
  
  return train_loss, train_acc




def validation_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):
  
  # Put model in eval mode
  model.eval() 
  
  # Setup validation loss and validation accuracy values
  validation_loss, validation_acc = 0, 0
  
  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)
  
          # 1. Forward pass
          validation_pred_logits = model(X)

          # 2. Calculate and accumulate loss
          loss = loss_fn(validation_pred_logits, y)
          validation_loss += loss.item()
          
          # Calculate and accumulate accuracy
          validation_pred_labels = validation_pred_logits.argmax(dim=1)
          validation_acc += ((validation_pred_labels == y).sum().item()/len(validation_pred_labels))
          
          if batch % 100 == 0:
            print(f"Batch {batch}/{len(dataloader)}")
            
  # Adjust metrics to get average loss and accuracy per batch 
  validation_loss = validation_loss / len(dataloader)
  validation_acc = validation_acc / len(dataloader)
  
  print(f"validation Loss: {validation_loss:.4f}, validation Acc: {validation_acc:.4f}")
  
  return validation_loss, validation_acc




