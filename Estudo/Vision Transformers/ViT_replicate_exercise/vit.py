import torch
from torch import nn
import torchvision


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_size, in_channels):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):

        x = self.conv(x)
        #print(f"Antes do reshape {x.shape}")
        x = self.flatten(x) 
        #print(f"Depois do reshape {x.shape}")
        
        x = x.permute(0, 2, 1)
        #print(f"Depois do reshape {x.shape}")
        
        return x
    
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        self.positional_embedding = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)
        #print(f"Positional Embedding shape: {self.positional_embedding.shape}")
    
    def forward(self, x):
        #print(f"X shape: {x.shape}")
        
        batch_size = x.shape[0]
        #print(f"Positional Embedding: {self.positional_embedding[:batch_size].shape}")
        x = x + self.positional_embedding[:batch_size]
        #print(f"X with Positional Embedding: {x.shape}")
        
        return x 
        
        
class ClassificationHead(nn.Module):
    def __init__(self, d_model, n_hidden, n_classes):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, n_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_classes)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x
        
        
        
class VisionTransformer(nn.Module):
    def __init__(self, n_layers, d_model, heads, patch_size, in_channels, n_classes, device):
        super().__init__()       
        
        self.patching_embedding = PatchEmbedding(d_model=d_model, patch_size=patch_size, in_channels=in_channels)
        
        self.positional_embeddings = PositionalEmbedding(d_model=d_model)
        
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)
        
        self.dropout = nn.Dropout(p=0.1)
        
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer=torch.nn.TransformerEncoderLayer(
                                                                                   d_model=d_model, 
                                                                                   nhead=heads, 
                                                                                   activation='gelu', 
                                                                                   batch_first=True, 
                                                                                   norm_first=True, 
                                                                                   device=device),
                                    num_layers=n_layers,
                                    enable_nested_tensor=False)            
        
        

        self.norm = nn.LayerNorm(d_model)
        
        
        self.mlp_head = ClassificationHead(d_model=d_model, n_hidden=2048, n_classes=n_classes).to(device)
        
    def forward(self, x):

        x = self.patching_embedding(x)
        
        x = self.positional_embeddings(x)
        
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)
        #print(f"X shape: {x.shape}")
        
        x = self.dropout(x)
        
        x = self.transformer_encoder(x)

        x = self.norm(x)
        
        x = self.mlp_head(x[:, 0])
        
        return x
        