"""
Variational Autoencoder (VAE) implementation for Kanji dataset using PyTorch and PyTorch Lightning.
Classes:
    KanjiDataset: Custom dataset class for loading Kanji images.
    Encoder: Convolutional encoder network for the VAE.
    Decoder: Convolutional decoder network for the VAE.
    LitVAE: LightningModule for training the VAE.
Functions:
    reparameterize(mu, logvar): Reparameterization trick to sample from N(mu, var) from N(0,1).
    forward(x): Forward pass through the VAE.
    training_step(batch, batch_idx): Training step for the VAE.
    configure_optimizers(): Configures the optimizer for training.
    sample(num_samples=9): Generates samples from the learned latent space.
    on_train_epoch_end(): Callback to save sample images at the end of each epoch.
Usage:
    1. Load the Kanji dataset using the KanjiDataset class.
    2. Create a DataLoader for the dataset.
    3. Initialize the LitVAE model.
    4. Set up the PyTorch Lightning Trainer with the desired configurations.
    5. Train the model using the Trainer's fit method.
"""

# Import necessary libraries and modules
import os
from PIL import Image
import sys

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import models
from torchvision.io import write_video
from torchvision.transforms import v2
import tqdm

# Set the default tensor type to float32
torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the custom dataset class for loading Kanji images
class KanjiDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = []
        for i, file in tqdm.tqdm(enumerate(os.listdir(root)), desc='Loading images'):
            img = Image.open(os.path.join(root, file))
            self.data.append(img)
        if transform:
            self.data = [self.transform(img) for img in self.data]

        self.data = torch.stack(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
# Load the dataset
transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

data = KanjiDataset('kanji', transform=transform)

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
        self.input1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.resnet = nn.Sequential(
            EncoderBlock(64, 64, stride=2), # 24x24x64
            EncoderBlock(64, 128),
            EncoderBlock(128, 128, stride=2), # 12x12x128
            EncoderBlock(128, 256),
            EncoderBlock(256, 256, stride=2), # 6x6x256
            EncoderBlock(256, 512),
            EncoderBlock(512, 512, stride=2), # 3x3x512
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 3 * 3, latent_dim)
        self.fc2 = nn.Linear(512 * 3 * 3, latent_dim)

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
        self.fc = nn.Linear(latent_dim, 512 * 3 * 3)
        self.unflatten = nn.Unflatten(1, (512, 3, 3))
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

# Define the LightningModule for training the VAE
class LitVAE(L.LightningModule):
    def __init__(self, latent_dim=32, lr=3e-3, sample_every=10, epochs=101):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim
        self.lr = lr
        self.sample_every = sample_every
        self.epochs = epochs
        self.save_hyperparameters()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z)
    
    def training_step(self, batch, batch_idx):
        x = batch
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)

        BCE = F.binary_cross_entropy_with_logits(x_hat, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = BCE + KLD
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        sch = OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.epochs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': sch,
                'interval': 'epoch'
            }
        }
    
    def sample(self, num_samples=9):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = torch.sigmoid(self.decoder(z))
        return samples
    
    def interpolate(self, num_frames=300):
        with torch.no_grad():
            start = Image.open('yui.png').convert('L')
            end = Image.open('joe.png').convert('L')
            start = transform(start).unsqueeze(0)
            end = transform(end).unsqueeze(0)
            mu_start, logvar_start = self.encoder(start)
            mu_end, logvar_end = self.encoder(end)
            
            alphas = torch.linspace(0, 1, num_frames)
            interpolations = []
            for alpha in alphas:
                mu = (1 - alpha) * mu_start + alpha * mu_end
                decoded = torch.sigmoid(self.decoder(mu))
                frame = (decoded * 255).type(torch.uint8)
                frame = frame.squeeze(0).permute(1, 2, 0).repeat(1, 1, 3)
                frame = 255 - frame
                interpolations.append(frame)
            return torch.stack(interpolations)
    
    def on_train_epoch_end(self):
        if self.current_epoch % self.sample_every == 0:
            samples = self.sample()
            grid = torchvision.utils.make_grid(samples, nrow=3)
            grid = (grid * 255).type(torch.uint8).cpu().numpy().transpose(1, 2, 0).squeeze()
            img = Image.fromarray(grid, mode='RGB')
            img = Image.fromarray(255 - np.array(img))
            os.makedirs('resnet_vae_samples', exist_ok=True)
            img.save(f'resnet_vae_samples/sample_{self.current_epoch}.png')

    def on_fit_end(self):
        write_video('resnet_interpolation.mp4', self.interpolate(), 30, video_codec='h264')

    def on_train_epoch_start(self):
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

# Hyperparameters
latent_dim = 1024
lr = 5e-3
sample_every = 5
epochs = 101
batch_size = 256


dataloader = DataLoader(
    data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory=True
)

vae = LitVAE(latent_dim=latent_dim, lr=lr, sample_every=sample_every, epochs=epochs)
checkpoint = L.pytorch.callbacks.ModelCheckpoint('resnet_vae_checkpoints', monitor='train_loss', save_top_k=1)
trainer = L.Trainer(
    max_epochs=epochs, 
    # callbacks=[checkpoint], 
    precision='bf16-mixed'
)
trainer.fit(vae, dataloader)