# A file to create a model
import torchvision
import torch
from torchinfo import summary
from torch import nn

def get_model():
    model = torchvision.models.efficientnet_b0(pretrained=True)


    # Para ver o modelo usamos "torchinfo" -> summary
    # pip install torchinfo
    batch_size = 16
    #summary(model, input_size=(batch_size, 3, 224, 224)) # batch_size, channels, height, width do input
    # Podemos ver que temos 100 classes de output


    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in model.features.parameters():
        param.requires_grad = False
        

    # Ver o nome da ultima Squential layer (classifier)
    #print(model)
    # Alterar a classifier layer para o num de outputs desejado (classes)
    model.classifier = torch.nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=100, bias=True)
    )


    # Verificar se o modelo foi alterado
    #summary(model, input_size=(batch_size, 3, 224, 224))

    return model
