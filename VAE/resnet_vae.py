# Import necessary libraries and modules
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Encoder resnet block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out

# Decoder resnet block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        output_padding = 1 if stride == 2 else 0
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, output_padding=output_padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, output_padding=output_padding),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out

# Define the encoder and decoder networks for the VAE
class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.final_size = 64 // 2**4
        self.input1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.resnet = nn.Sequential(
            EncoderBlock(64, 64, stride=2), 
            EncoderBlock(64, 128),
            EncoderBlock(128, 128, stride=2), 
            EncoderBlock(128, 256),
            EncoderBlock(256, 256, stride=2), 
            EncoderBlock(256, 512),
            EncoderBlock(512, 512, stride=2), 
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * self.final_size**2, latent_dim)
        self.fc2 = nn.Linear(512 * self.final_size**2, latent_dim)

    def forward(self, x):
        x = self.input1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.resnet(x)
        x = self.flatten(x)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.init_size = 64 // 2**4
        self.fc = nn.Linear(latent_dim, 512 * self.init_size**2)
        self.unflatten = nn.Unflatten(1, (512, self.init_size, self.init_size))
        self.conv_block = nn.Sequential(
            DecoderBlock(512, 512, stride=2), # 6x6x512
            DecoderBlock(512, 256),
            DecoderBlock(256, 256, stride=2), # 12x12x256
            DecoderBlock(256, 128),
            DecoderBlock(128, 128, stride=2), # 24x24x128
            DecoderBlock(128, 64),
            DecoderBlock(64, 64, stride=2) # 48x48x64
        )
        self.output1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.output2 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.conv_block(x)
        x = self.output1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.output2(x)
        return x