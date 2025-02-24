# Import necessary libraries and modules
import argparse
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

import gan
import resgan

parser = argparse.ArgumentParser(description='Train a generative adversarial network on Kanji characters')
parser.add_argument('--batch_size', type=int, default=64, help='The batch size for training')
parser.add_argument('--epochs', type=int, default=300, help='The number of epochs to train for')
parser.add_argument('--latent_dim', type=int, default=256, help='The dimension of the latent space')
parser.add_argument('--lr', type=float, default=4e-4, help='The learning rate for training')
parser.add_argument('--save_dir', type=str, default='gan', help='The directory to save the model and logs')
parser.add_argument('--sample_every', type=int, default=5, help='The number of epochs between sampling')
parser.add_argument('--model', type=str, default='resnet', help='The model architecture to use (resnet or conv)')
parser.add_argument('--loss', type=str, default='hinge', choices=['hinge', 'wasserstein', 'BCE'], help='The loss function to use (BCE, hinge, or wasserstein)')
parser.add_argument('--gp', type=float, default=10, help='The gradient penalty coefficient')
parser.add_argument('--layers', nargs='+', type=int, default=[512, 256, 128, 64], help='The number of channels in each layer of the generator')
args = parser.parse_args()

device  = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    v2.Resize((64,64)),
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

dataloader = DataLoader(
    KanjiDataset('kanji', transform=transform),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory=True
)

class GAN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        if args.model == 'resnet':
            self.generator = resgan.Generator(args.latent_dim, args.layers)
            self.discriminator = resgan.Discriminator(args.layers[::-1])
        else:
            self.generator = gan.Generator(args.latent_dim, args.layers)
            self.discriminator = gan.Discriminator(args.layers[::-1])

        if args.loss == 'hinge':
            self.d_loss_fn = self._hinge_loss
        elif args.loss == 'wasserstein':
            self.d_loss_fn = self._wasserstein_loss
        else:
            self.d_loss_fn = self._bce_loss

    def _hinge_loss(self, real_pred, fake_pred, grad_penalty):
        return F.relu(1 - real_pred).mean() + F.relu(1 + fake_pred).mean()
    def _wasserstein_loss(self, real_pred, fake_pred, grad_penalty):
        return -real_pred.mean() + fake_pred.mean()
    def _bce_loss(self, real_pred, fake_pred, grad_penalty):
        return F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)) + F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
    
    def _gradient_penalty(self, real_img, fake_img):
        alpha = torch.rand(real_img.size(0), 1, 1, 1, device=device)
        interpolated = alpha * real_img + (1 - alpha) * fake_img
        interpolated.requires_grad = True
        pred = self.discriminator(interpolated).squeeze()
        gradients = torch.autograd.grad(
            outputs=pred, 
            inputs=interpolated, 
            grad_outputs=torch.ones_like(pred), 
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        return args.gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def forward(self, x):
        return self.generator(x)
    
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        real_img = batch

        # Train the discriminator 5 times
        for _ in range(5):
            opt_d.zero_grad()
            latent = torch.randn(real_img.size(0), args.latent_dim, device=device)
            with torch.no_grad():
                fake_img = self(latent)

            grad_penalty = self._gradient_penalty(real_img, fake_img) if args.loss != 'BCE' else None

            real_pred = self.discriminator(real_img).squeeze()
            fake_pred = self.discriminator(fake_img).squeeze()
            

            d_loss = self.d_loss_fn(real_pred, fake_pred, grad_penalty)
            self.manual_backward(d_loss)
            opt_d.step()

        self.sch_d.step()

        # Train the generator
        opt_g.zero_grad()
        opt_d.zero_grad()
        latent = torch.randn(real_img.size(0), args.latent_dim, device=device)
        fake_img = self(latent)
        fake_pred = self.discriminator(fake_img).squeeze()
        if args.loss == 'BCE':
            g_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
        else:
            g_loss = -fake_pred.mean()
        self.manual_backward(g_loss)
        opt_g.step()
        self.sch_g.step()

        self.log_dict({'d_loss': d_loss, 'g_loss': g_loss}, prog_bar=True)

    def configure_optimizers(self):
        lr = args.lr
        opt_g = Adam(self.generator.parameters(), lr=lr)
        opt_d = Adam(self.discriminator.parameters(), lr=lr)
        self.sch_g = OneCycleLR(opt_g, max_lr=lr, total_steps=self.trainer.estimated_stepping_batches)
        self.sch_d = OneCycleLR(opt_d, max_lr=lr, total_steps=self.trainer.estimated_stepping_batches)
        return [opt_g, opt_d], []
    
    def sample(self, num_samples=9):
        with torch.no_grad():
            self.eval()
            latent = torch.randn(num_samples, args.latent_dim, device=device)
            return self(latent)
        
    def on_train_epoch_end(self):
        if (self.current_epoch+1) % args.sample_every == 0:
            grid = torchvision.utils.make_grid(self.sample(), nrow=3, normalize=True)
            grid = 255 - grid * 255
            grid = grid.type(torch.uint8).cpu().numpy().transpose(1,2,0).squeeze()
            img = Image.fromarray(grid, mode='RGB')
            os.makedirs(f'{args.save_dir}_{args.model}_samples', exist_ok=True)
            img.save(f'{args.save_dir}_{args.model}_samples/sample_{self.current_epoch}.png')

gan = GAN()
trainer = L.Trainer(
    max_epochs=args.epochs,
    precision='bf16-mixed',
    default_root_dir=f'{args.save_dir}_{args.model}_dim{args.latent_dim}_layers{args.layers}_{args.epochs}',
)

trainer.fit(gan, dataloader)