import lightning as L
import torch

from main import GAN

model = GAN.load_from_checkpoint('gan_resnet_kanji_dim128_layers[512, 256, 128]/lightning_logs/version_5/checkpoints/epoch=99-step=48600.ckpt')

gen = model.generator
torch.save(gen.state_dict(), 'GAN/resnet_ganji_gen.pth')