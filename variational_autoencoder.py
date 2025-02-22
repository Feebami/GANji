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

# Define the encoder and decoder networks for the VAE
class Encoder(nn.Module):
    def __init__(self, latent_dim=32, filter_list=[16, 32, 64]):
        super().__init__()
        final_size = int(48 * 2**-len(filter_list))
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
    def __init__(self, latent_dim=32, filter_list=[64, 32, 16]):
        super().__init__()
        final_size = int(48 * 2**-len(filter_list))
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

# Define the LightningModule for training the VAE
class LitVAE(L.LightningModule):
    def __init__(self, latent_dim=32, filter_list=[16, 32, 64], lr=3e-3, sample_every=10, max_epochs=100):
        super().__init__()
        self.encoder = Encoder(latent_dim, filter_list)
        self.decoder = Decoder(latent_dim, filter_list[::-1])
        self.latent_dim = latent_dim
        self.lr = lr
        self.sample_every = sample_every
        self.max_epochs = max_epochs
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        sch = OneCycleLR(optimizer, max_lr=self.lr, total_steps=101)
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
            os.makedirs('samples', exist_ok=True)
            img.save(f'samples/sample_{self.current_epoch}.png')

    def on_fit_end(self):
        write_video('interpolation.mp4', self.interpolate(), 60, video_codec='h264')

    def on_train_epoch_start(self):
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

# Hyperparameters
latent_dim = 512
lr = 1e-2
sample_every = 5
filter_list = [32, 64, 128, 256]
max_epochs = 101
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

vae = LitVAE(latent_dim=latent_dim, filter_list=filter_list, lr=lr, sample_every=sample_every, max_epochs=max_epochs)
checkpoint = L.pytorch.callbacks.ModelCheckpoint('vae_checkpoints', monitor='train_loss', save_top_k=1)
trainer = L.Trainer(
    max_epochs=max_epochs, 
    callbacks=[checkpoint], 
    precision='bf16-mixed'
)
trainer.fit(vae, dataloader)