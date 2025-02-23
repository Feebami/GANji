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
from torch.nn.utils import spectral_norm
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
    v2.Resize((48,48)),
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

data = KanjiDataset('kanji', transform=transform)

# Define the generator model
class Generator(nn.Module):
    def __init__(self, latent_dim=128, layer_list=[256, 128, 64]):
        super().__init__()
        self.init_size = 48 // 2**len(layer_list)
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
            nn.ConvTranspose2d(layer_list[-1], 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
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
    def __init__(self, img_channels=1, layer_list=[64, 128, 256]):
        super().__init__()
        self.final_size = 48 // 2**len(layer_list)
        self.model = nn.Sequential()
        self.model.add_module('input', nn.Conv2d(img_channels, layer_list[0], 4, stride=2, padding=1, bias=False))
        for i in range(len(layer_list) - 1):
            self.model.add_module(f'block{i}', nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(layer_list[i], layer_list[i+1], 4, stride=2, padding=1, bias=False),
            ))
        self.output = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(layer_list[-1] * self.final_size**2, 1)
        )
        
    def forward(self, x):
        x = self.model(x)
        x = self.output(x)
        return x
    
# Define the GAN model
class GAN(L.LightningModule):
    def __init__(self, latent_dim=128, g_layers=[256,128,64], d_layers=[64,128,256], lr=3e-4, sample_every=10):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = Generator(latent_dim, g_layers)
        self.discriminator = Discriminator(layer_list=d_layers)
        self.d_loss = np.inf

    def forward(self, x):
        return self.generator(x)
    
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        real_imgs = batch

        latent = torch.randn(real_imgs.size(0), self.hparams.latent_dim, device=device)

        # Train the discriminator
        if self.trainer.global_step % 5 == 0:
            opt_d.zero_grad()

            with torch.no_grad():
                fake_imgs = self.generator(latent)

            real_loss = F.binary_cross_entropy_with_logits(self.discriminator(real_imgs).squeeze(), torch.ones(real_imgs.size(0), device=device))
            fake_loss = F.binary_cross_entropy_with_logits(self.discriminator(fake_imgs).squeeze(), torch.zeros(real_imgs.size(0), device=device))
            self.d_loss = real_loss + fake_loss

            self.manual_backward(self.d_loss)
            opt_d.step()

        self.sch_d.step()

        # Train the generator
        opt_g.zero_grad()
        fake_imgs = self.generator(latent)
        g_loss = F.binary_cross_entropy_with_logits(self.discriminator(fake_imgs).squeeze(), torch.ones(real_imgs.size(0), device=device))
        self.manual_backward(g_loss)
        opt_g.step()
        self.sch_g.step()

        self.log_dict({'d_loss': self.d_loss, 'g_loss': g_loss}, prog_bar=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = Adam(self.generator.parameters(), lr=lr)
        opt_d = Adam(self.discriminator.parameters(), lr=lr*2)
        self.sch_g = OneCycleLR(opt_g, max_lr=lr, total_steps=self.trainer.estimated_stepping_batches)
        self.sch_d = OneCycleLR(opt_d, max_lr=lr, total_steps=self.trainer.estimated_stepping_batches)
        return [opt_g, opt_d], []
    
    def sample(self, num_samples=9):
        with torch.no_grad():
            self.generator.eval()
            latent = torch.randn(num_samples, self.hparams.latent_dim, device=device)
            samples = self.generator(latent)
        return samples
    
    def on_train_epoch_end(self):
        if self.current_epoch % self.hparams.sample_every == 0:
            samples = self.sample()
            grid = torchvision.utils.make_grid(samples, nrow=3, normalize=True)
            grid = 255 - grid * 255
            grid = grid.type(torch.uint8).cpu().numpy().transpose(1,2,0).squeeze()
            img = Image.fromarray(grid, mode='RGB')
            os.makedirs('gan_samples', exist_ok=True)
            img.save(f'gan_samples/sample_{self.current_epoch}.png')

# Hyperparameters
latent_dim = 128
g_layers = [32, 16]
d_layers = [16, 32]
lr = 3e-4
sample_every = 10
epochs = 201
batch_size = 512

# Data loader
dataloader = DataLoader(
    data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory=True
)

gan = GAN(latent_dim=latent_dim, g_layers=g_layers, d_layers=d_layers, lr=lr, sample_every=sample_every)
checkpoint = L.pytorch.callbacks.ModelCheckpoint(
    dirpath='gan_checkpoints',
    filename='gan-{epoch:02d}',
    every_n_epochs=5,
    save_top_k=-1,
)
trainer = L.Trainer(
    max_epochs=epochs,
    callbacks=[checkpoint],
    precision='bf16-mixed',
)
trainer.fit(gan, dataloader)