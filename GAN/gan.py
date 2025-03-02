# Import necessary libraries and modules
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the generator model
class Generator(nn.Module):
    def __init__(self, latent_dim=128, layer_list=[256, 128, 64], img_channels=1):
        super().__init__()
        self.init_size = 64 // 2**len(layer_list)
        self.fc = nn.Linear(latent_dim, layer_list[0] * self.init_size ** 2, bias=False)
        self.unflatten = nn.Unflatten(1, (layer_list[0], self.init_size, self.init_size))
        self.blocks = nn.ModuleList()
        for i in range(len(layer_list) - 1):
            self.blocks.append(
                nn.Sequential(
                    nn.BatchNorm2d(layer_list[i]),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(layer_list[i], layer_list[i+1], 4, stride=2, padding=1, bias=False),
                )
            )
        self.final = nn.Sequential(
            nn.BatchNorm2d(layer_list[-1]),
            nn.ReLU(True),
            nn.ConvTranspose2d(layer_list[-1], img_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        
    
    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        for block in self.blocks:
            x = block(x)
        x = self.final(x)
        return x
    
# Define the discriminator model
class Discriminator(nn.Module):
    def __init__(self, layer_list=[64, 128, 256], img_channels=1):
        super().__init__()
        self.final_size = 64 // 2**len(layer_list)
        self.model = nn.Sequential()
        self.model.add_module('input', nn.Conv2d(img_channels, layer_list[0], 4, stride=2, padding=1, bias=False))
        for i in range(len(layer_list) - 1):
            self.model.add_module(f'block{i}', nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(layer_list[i], layer_list[i+1], 4, stride=2, padding=1, bias=False),
            ))
        self.output = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(layer_list[-1] * self.final_size**2, 1)
        )
        
    def forward(self, x):
        x = self.model(x)
        x = self.output(x)
        return x