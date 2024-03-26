import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
from patchifying import patchify, get_positional_embeddings
from multi_head_self_attention import MyMSA

np.random.seed(0)
torch.manual_seed(0)


def model_train_test(MyViT):
    batch_size=32
    
    train_data = MNIST(root='data', train=True, transform=ToTensor(), download=True)
    test_data = MNIST(root='data', train=False, transform=ToTensor(), download=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    
    model = MyViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    
    N_EPOCHS = 5
    LR = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
        
        
class MyViT(nn.Module):
    def __init__(self, chw=(1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        super(MyViT, self).__init__()
        
        # Attributes
        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Check if input shape is divisible by n_patches
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        
        # Para saber o tamanho de cada patch (H * W)
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches) # (H / n_patches, W / n_patches)

    # 1) Linear mapper
        # Determina o shape (C * H * W) do input (cada patch) 
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1]) 
        # Cada patch é mapeada para um vetor de tamanho hidden_d (parametro do modelo)
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

    # 2) Learnable classifiation token
        # Criar um parametro que vai ser atualizado durante o train
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d)) # shape para cada patch

    # 3) Positional embedding
        # Define as posições dos patches na imagem original
        self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d)))
        self.pos_embed.requires_grad = False
        
    # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
    # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )    
    
    def forward(self, images):

        # Recebe o num de imagens do batch e o num de patchs desejado para cada img
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches) # return the patches (sub-imgs) for each image in the batch (images)
        print(patches.shape+" -> N images, P patches, X values for each pixel of each patch (px X px) = X") 
        
        # Mapeia cada patch para um vetor de tamanho hidden_d
        tokens = self.linear_mapper(patches)
        print(tokens.shape+" -> N images, P patches, X turn into a vector of size hidden_d = X with the linear layer") 
        
        # Percorre cada patch e adiciona verticalmente (vstack) o class_token (classification token), exemplo:
        # [[class_token],
        # [token[i]]]
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        
        # Adding positional embedding
        # n -> num de imagens no batch (para cada imagem no batch determina a posição de cada patch)
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        
        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
        
        # Getting the classification token only
        out = out[:, 0]
        
        return self.mlp(out) # Map to output dimension, output category distribution
        



class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        
        
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        
        out = out + self.mlp(self.norm2(out))
        return out


    
if __name__ == '__main__':
  # Current model
  model = MyViT(
    chw=(1, 28, 28),
    n_patches=7
  )
    # 7 images of 28x28 pixels with 1 channel
  x = torch.randn(7, 1, 28, 28) # Dummy images
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = MyViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
  
    
