import torch
from torchinfo import summary
from torch import nn


class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """
    # Initialize the class with appropriate variables
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()
        
        self.patch_size = patch_size

        # Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # Define the forward method
    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # 6. Make sure the output shape has the right order
        
        #print(f"Output shape from patching and embbeding: {x_flattened.shape} (flatened, patched and embedded)")
        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]  
    
    
    
    
class ViT(nn.Module):
    def __init__(self, img_size:int=224, in_channels:int=3, patch_size:int=16, embedding_dim:int=768, 
                 mlp_size:int=3072, heads:int=12, num_encoder_layers:int=12, device:str="cuda", 
                 dropout=0.1, num_classes=3):  # VALUES FROM THE PAPER
        super().__init__()
        
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size."
        
        
        # Create patch embedding layer
        self.patch_embedder = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim)
        
        
        # Class token
        # Create a learnable parameter for the class token
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim)) 
        
        
        # Create position embedding
        num_patches = (img_size * img_size) // patch_size**2 # N = HW/P^2
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches+1, embedding_dim)) # +1 for the class token
        
        
        # Create patch + position embedding dropout
        self.embedding_dropout = nn.Dropout(p=dropout)
        

        # Create Transformer Encoder (Single Block) -> initialize in the encoder method below
        """ self.transformer_encoder_layer = nn.TransformerEncoderLayer(
                                                           d_model=embedding_dim,     # By paper = 768
                                                           nhead=heads,               # By paper = 12
                                                           dim_feedforward=mlp_size,  # By paper = 3072
                                                           activation="gelu",         # By paper
                                                           batch_first=True,          # Batch value comes first in the input shape
                                                           norm_first=True,           # Norm layer comes in first in the block (paper)
                                                           device=device), """
        
        
        # Create Stack of Transformer Encoder Layers
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(              # Create a single block
                                                           d_model=embedding_dim,     # By paper = 768
                                                           nhead=heads,               # By paper = 12
                                                           dim_feedforward=mlp_size,  # By paper = 3072
                                                           activation="gelu",         # By paper
                                                           batch_first=True,          # Batch value comes first in the input shape
                                                           norm_first=True,           # Norm layer comes in first in the block (paper)
                                                           device=device),            
                                            num_encoder_layers,
                                            enable_nested_tensor=False) # Stack the single block num_encoder_layers times
        
        
        # Create MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

        
    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        batch_size = x.shape[0]
        
    # PATCHING AND EMBEDDING
        patch_embeddings = self.patch_embedder(x)
        #print(f"Patching, embeddings and flatten shape: {patch_embeddings.shape} (Batch, N_patches, P^2•C)")
        
    # CLASS TOKEN
        # Prepare creating a tensor (token) with the same batch size to add to the embeddings
        class_token = self.class_token.expand(batch_size, -1, -1) # -1 means keep the same dimension
        x = torch.cat([class_token, patch_embeddings], dim=1) # Add the class token to the embeddings in dim 1 to each image
        #print(f"Add class token with shape: {class_token.shape} to each image (batch)")
        #print(f"After adding class token: {x.shape} (each image has one class token and N patches(796))")
        
    # POSITION EMBEDDING
        # Add the class token to the embeddings (for each patch)
        x = self.position_embedding + x # Add the position embedding to the embeddings in dim 1
        #print(f"Added position embedding shape: {x.shape} (same)")
        
    # EMBEDDING DROPOUT
        x = self.embedding_dropout(x)
        #print(f"Passed by dropout: {x.shape}")
        
    # TRANSFORMER ENCODER
        x = self.transformer_encoder(x)
        #print(f"Passed by transformer encoder: {x.shape}")

    # MLP HEAD (Pass Class Token (0th index) through MLP)
        x = self.mlp_head(x[:, 0])
        #print(f"Passed by MLP head (The class token of each image only [:, 0]): {x}")
        
        return x
        
        
        
        
        
        
        
        
        
        
        