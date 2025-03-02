import lightning as L
import torch

from main import VAE

model = VAE.load_from_checkpoint('vae_resnet_dim256_100/lightning_logs/version_4/checkpoints/epoch=99-step=4100.ckpt')

decoder = model.decoder
torch.save(decoder.state_dict(), 'models/resnet_ganji_decoder.pth')