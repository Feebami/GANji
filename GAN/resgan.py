# Import necessary libraries and modules
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        output_padding = 1 if stride == 2 else 0
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride, padding=1, output_padding=output_padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 1, stride, output_padding=output_padding, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)
    
class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
            )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

# Define the generator model
class Generator(nn.Module):
    def __init__(self, latent_dim=128, layer_list=[256, 128, 64], img_channels=1):
        super().__init__()
        self.init_size = 64 // 2**len(layer_list)
        self.input = nn.Sequential(
            nn.Linear(latent_dim, layer_list[0] * self.init_size ** 2, bias=False),
            nn.Unflatten(1, (layer_list[0], self.init_size, self.init_size))
        )
        self.blocks = nn.ModuleList()
        for i in range(len(layer_list) - 1):
            self.blocks.append(GenBlock(layer_list[i], layer_list[i+1]))
        self.output = nn.Sequential(
            GenBlock(layer_list[-1], layer_list[-1]),
            GenBlock(layer_list[-1], img_channels, stride=1),
            nn.Tanh(),
        )
        self.model = nn.Sequential(
            self.input,
            *self.blocks,
            self.output
        )
        self._init_weights(self.model)

    def _init_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1.0)
    
    def forward(self, x):
        return self.model(x)
    
# Define the discriminator model
class Discriminator(nn.Module):
    def __init__(self, layer_list=[64, 128, 256], img_channels=1):
        super().__init__()
        self.final_size = 64 // 2**len(layer_list)
        self.input = nn.Sequential(
            DisBlock(img_channels, layer_list[0], stride=1),
            DisBlock(layer_list[0], layer_list[0]),
        )
        self.blocks = nn.ModuleList()
        for i in range(1, len(layer_list)):
            self.blocks.append(DisBlock(layer_list[i-1], layer_list[i]))
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(layer_list[-1] * self.final_size**2, 1),
        )
        self.model = nn.Sequential(
            self.input,
            *self.blocks,
            self.output
        )
        self._init_weights(self.model)

    def _init_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d)or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1.0)


    def forward(self, x):
        return self.model(x)