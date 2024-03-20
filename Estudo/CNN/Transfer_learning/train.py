# A file to leverage all other files and train a target model.

import torch
import data_setup, engine, model, utils
from tqdm import tqdm

# Setup hyperparameters
EPOCHS = 5
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001


# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Create DataLoaders with help from data_setup.py
train_dataloader, val_dataloader = data_setup.create_cat_dog_dataloaders()


model = model.get_model().to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch+1}/{EPOCHS}")
    
    train_loss, train_acc = engine.train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      
    test_loss, test_acc = engine.test_step(model=model,
                                        dataloader=val_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
      


# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="model_exercise_transfer_learning.pth")