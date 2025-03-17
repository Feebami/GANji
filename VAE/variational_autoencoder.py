# Import necessary libraries and modules
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the encoder and decoder networks for the VAE
class Encoder(nn.Module):
    def __init__(self, latent_dim, filter_list=[32, 64, 128]):
        super().__init__()
        final_size = int(64 * 2**-len(filter_list))
        self.conv_block = nn.ModuleList()
        for i, filters in enumerate(filter_list):
            if i == 0:
                self.conv_block.append(nn.Conv2d(1, filters, 3, stride=2, padding=1))
                self.conv_block.append(nn.Conv2d(filters, filters, 3, padding=1))
            else:
                self.conv_block.append(nn.Conv2d(filter_list[i-1], filters, 3, stride=2, padding=1))
                self.conv_block.append(nn.Conv2d(filters, filters, 3, padding=1))
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(filter_list[-1] * final_size**2, latent_dim)
        self.fc2 = nn.Linear(filter_list[-1] * final_size**2, latent_dim)

    def forward(self, x):
        for layer in self.conv_block:
            x = F.relu(layer(x))
        x = self.flatten(x)
        lat1 = self.fc1(x)
        lat2 = self.fc2(x)
        return lat1, lat2
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=32, filter_list=[128, 64, 32]):
        super().__init__()
        final_size = int(64 * 2**-len(filter_list))
        self.fc = nn.Linear(latent_dim, filter_list[0] * final_size**2)
        self.unflatten = nn.Unflatten(1, (filter_list[0], final_size, final_size))
        self.conv_block = nn.ModuleList()
        for i in range (len(filter_list) - 1):
            self.conv_block.append(nn.ConvTranspose2d(filter_list[i], filter_list[i+1], 3, stride=2, padding=1, output_padding=1))
            self.conv_block.append(nn.Conv2d(filter_list[i+1], filter_list[i+1], 3, padding=1))
        self.output1 = nn.Conv2d(filter_list[-1], filter_list[-1], 3, padding=1)
        self.output2 = nn.ConvTranspose2d(filter_list[-1], 1, 3, stride=2, padding=1, output_padding=1)
                

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.unflatten(x)
        for layer in self.conv_block:
            x = F.relu(layer(x))
        x = F.relu(self.output1(x))
        x = self.output2(x)
        return x