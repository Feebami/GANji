from PIL import Image
import os

import lightning as L
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm

import unet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

img_channels = 1

class Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.h_flip = v2.RandomHorizontalFlip()
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
        return self.h_flip(self.data[idx])
    
class DDPM(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.beta = self._cosine_beta().to(device)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, 0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_1m_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod)

        self.model = unet.UNet(img_channels)

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
        t = torch.randint(0, 1000, (x.size(0),), device=device)
        x_t, noise = self.add_noise(x, t)
        noise_hat = self.model(x_t, t)
        loss = F.mse_loss(noise_hat, noise)
        self.log('loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=1e-4)
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
            for i in reversed(range(1000)):
                t = torch.full((n,), i, device=device)
                alpha = self.alpha[t].view(-1, 1, 1, 1)
                sqrt_1m_alpha_cumprod = self.sqrt_1m_alpha_cumprod[t].view(-1, 1, 1, 1)
                beta = self.beta[t].view(-1, 1, 1, 1)
                noise_cumprod = self.model(x, t)
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (x - (beta/sqrt_1m_alpha_cumprod) * noise_cumprod) / torch.sqrt(alpha) + torch.sqrt(beta) * noise
            return x
        
    def on_train_epoch_end(self):
        if self.current_epoch % 2 == 0:
            samples = self.sample(9)
            samples = samples * 0.5 + 0.5
            grid = torchvision.utils.make_grid(samples, nrow=3) * 255
            grid = grid.permute(1, 2, 0).cpu().numpy().astype('uint8')
            grid = Image.fromarray(grid)
            os.makedirs('ddpm_samples', exist_ok=True)
            grid.save(f'ddpm_samples/sample_{self.current_epoch}.png')

    def on_fit_end(self):
        self.model.to(device)
        samples = self.sample(9)
        samples = samples * 0.5 + 0.5
        grid = torchvision.utils.make_grid(samples, nrow=3) * 255
        grid = grid.permute(1, 2, 0).cpu().numpy().astype('uint8')
        grid = Image.fromarray(grid)
        os.makedirs('ddpm_samples', exist_ok=True)
        grid.save(f'ddpm_samples/sample_{self.current_epoch}.png')

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
        batch_size=16,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True
    )
    
    unet = DDPM()
    trainer = L.Trainer(
        max_epochs=100,
        default_root_dir='ddpm_logs',
        precision='bf16-mixed'
    )
    trainer.fit(unet, dataloader)