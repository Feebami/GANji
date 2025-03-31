import argparse
from PIL import Image
import os
import sys
import time

from cleanfid import fid
import lightning as L
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm

import unet

parser = argparse.ArgumentParser(description='Train a DDPM on Kanji characters')
parser.add_argument('--batch_size', type=int, default=32, help='The batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to train for')
parser.add_argument('--lr', type=float, default=5e-5, help='The learning rate for training')
parser.add_argument('--save_dir', type=str, default='ddpm', help='The directory to save the model and logs')
parser.add_argument('--sample_every', type=int, default=4, help='The number of epochs between sampling')
parser.add_argument('--n_steps', type=int, default=1024, help='The number of steps in the diffusion process')
parser.add_argument('--cosine_beta', action='store_true', help='Use a cosine beta schedule')
parser.add_argument('--attention', action='store_true', help='Use attention in the UNet')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

img_channels = 1

class Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = []
        for file in tqdm(os.listdir(root)):
            img = Image.open(os.path.join(root, file))
            if self.transform:
                img = self.transform(img)
            self.data.append(img)
        self.data = torch.stack(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
class DDPM(L.LightningModule):
    def __init__(self, n_steps=1000, cosine_beta=True, attention=True):
        super().__init__()
        self.n_steps = n_steps
        beta = self._cosine_beta(n_steps) if cosine_beta else torch.linspace(1e-4, 0.02, n_steps)
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', 1. - self.beta)
        self.register_buffer('alpha_cumprod', torch.cumprod(self.alpha, 0))
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(self.alpha_cumprod))
        self.register_buffer('sqrt_1m_alpha_cumprod', torch.sqrt(1. - self.alpha_cumprod))

        self.model = unet.UNet(img_channels, n_steps, attention)

    def _cosine_beta(self, n_steps=1000, s=8e-3):
        timesteps = torch.arange(n_steps+1, dtype=torch.float32) / n_steps + s
        alphas = timesteps / (1 + s) * 0.5 * torch.pi
        alphas = torch.cos(alphas) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
        return betas

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1m_alpha_cumprod = self.sqrt_1m_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod * x + sqrt_1m_alpha_cumprod * noise, noise

    def training_step(self, batch, batch_idx):
        x = batch
        t = torch.randint(0, self.n_steps, (x.size(0),), device=device)
        x_t, noise = self.add_noise(x, t)
        noise_hat = self.model(x_t, t)
        loss = F.mse_loss(noise_hat, noise)
        self.log('loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 100)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
    
    def sample(self, n):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(n, img_channels, 64, 64, device=device)
            for i in reversed(range(self.n_steps)):
                t = torch.full((n,), i, device=device)
                alpha = self.alpha[t].view(-1, 1, 1, 1)
                sqrt_1m_alpha_cumprod = self.sqrt_1m_alpha_cumprod[t].view(-1, 1, 1, 1)
                beta = self.beta[t].view(-1, 1, 1, 1)
                noise_hat = self.model(x, t)
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (x - (beta/sqrt_1m_alpha_cumprod) * noise_hat) / torch.sqrt(alpha) + torch.sqrt(beta) * noise
            return x
        
    def on_train_epoch_end(self):
        if self.current_epoch % args.sample_every == 0:
            sample_dir = f'{args.save_dir}_cosinebeta{args.cosine_beta}_attention{args.attention}_samples'
            samples = self.sample(9)
            samples = samples * 0.5 + 0.5
            samples = torch.clamp(samples, 0, 1)
            grid = torchvision.utils.make_grid(samples, nrow=3) * 255
            grid = grid.permute(1, 2, 0).cpu().numpy().astype('uint8')
            grid = Image.fromarray(grid)
            os.makedirs(sample_dir, exist_ok=True)
            grid.save(f'{sample_dir}/sample_{self.current_epoch}.png')

    def on_fit_end(self):
        self.to(device)
        for name, buffer in self.named_buffers():
            setattr(self, name, buffer.to(device))
        ddpm_dir = f'{args.save_dir}_cosinebeta{args.cosine_beta}_attention{args.attention}_score_imgs'
        os.makedirs(ddpm_dir, exist_ok=True)
        for i in tqdm(range(645)):
            samples = self.sample(16)
            samples = samples * 0.5 + 0.5
            samples = torch.clamp(samples, 0, 1)
            samples = 255 - samples * 255
            samples = v2.Resize((48, 48))(samples)
            samples = samples.type(torch.uint8).cpu().numpy().transpose(0, 2, 3, 1)
            for j, sample in enumerate(samples):
                img = Image.fromarray(sample.squeeze(), mode='L')
                img.save(os.path.join(ddpm_dir, f'sample_{i*16+j}.png'))
        fid_score = fid.compute_fid(ddpm_dir, 'kanji', device=device)
        print(f'FID score: {fid_score}')

if __name__ == '__main__':
    transform = v2.Compose([
        v2.Resize((64,64)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5]*img_channels, [0.5]*img_channels),
    ])
    data = Dataset('kanji', transform)
    dataloader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True
    )
    
    unet = DDPM(n_steps=args.n_steps, cosine_beta=args.cosine_beta, attention=args.attention)
    trainer = L.Trainer(
        max_epochs=args.epochs,
        default_root_dir='ddpm_logs',
        precision='bf16-mixed'
    )
    start_time = time.time()
    trainer.fit(unet, dataloader)
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total training and sampling time: {total_time:.2f} seconds')