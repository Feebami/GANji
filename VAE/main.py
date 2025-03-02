# Import necessary libraries and modules
import argparse
import os
from PIL import Image

import lightning as L
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import v2
import tqdm

import resnet_vae
import variational_autoencoder

parser = argparse.ArgumentParser(description='Train a variational autoencoder on Kanji characters')
parser.add_argument('--batch_size', type=int, default=256, help='The batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to train for')
parser.add_argument('--latent_dim', type=int, default=256, help='The dimension of the latent space')
parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate for training')
parser.add_argument('--save_dir', type=str, default='vae', help='The directory to save the model and logs')
parser.add_argument('--sample_every', type=int, default=10, help='The number of epochs between sampling')
parser.add_argument('--model', type=str, default='resnet', help='The model architecture to use (resnet or conv)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the custom dataset class for loading Kanji images
class KanjiDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.h_flip = v2.RandomHorizontalFlip()
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
        return self.h_flip(self.data[idx])
    
class VAE(L.LightningModule):
    def __init__(self):
        super().__init__()
        if args.model == 'resnet':
            self.encoder = resnet_vae.Encoder(latent_dim=args.latent_dim)
            self.decoder = resnet_vae.Decoder(latent_dim=args.latent_dim)
        else:
            self.encoder = variational_autoencoder.Encoder(latent_dim=args.latent_dim)
            self.decoder = variational_autoencoder.Decoder(latent_dim=args.latent_dim)
        
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
        optimizer = Adam(self.parameters(), lr=args.lr)
        sch = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': sch,
                'interval': 'epoch'
            }
        }
    
    def sample(self, num_samples=9):
        with torch.no_grad():
            z = torch.randn(num_samples, args.latent_dim, device=device)
            return self.decoder(z)
    
    def on_train_epoch_end(self):
        if (self.current_epoch+1) % args.sample_every == 0:
            samples = torch.sigmoid(self.sample())
            grid = torchvision.utils.make_grid(samples, nrow=3)
            grid = 255 - grid * 255
            grid = grid.type(torch.uint8).cpu().numpy().transpose(1,2,0).squeeze()
            img = Image.fromarray(grid, mode='RGB')
            os.makedirs(f'{args.save_dir}_{args.model}_samples', exist_ok=True)
            img.save(f'{args.save_dir}_{args.model}_samples/sample_{self.current_epoch}.png')

if __name__ == '__main__':

    transform = v2.Compose([
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

    vae = VAE()
    trainer = L.Trainer(
        max_epochs=args.epochs,
        precision='bf16-mixed' if device == 'cuda' else 32,
        default_root_dir=f'{args.save_dir}_{args.model}_dim{args.latent_dim}_{args.epochs}',
    )

    trainer.fit(vae, dataloader)