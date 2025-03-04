import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.emb_layer = nn.Linear(emb_dim, out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t):
        return self.block(x) + self.shortcut(x) + self.emb_layer(t).unsqueeze(-1).unsqueeze(-1)
    
class UNet(nn.Module):
    def __init__(self, img_channels):
        super().__init__()
        self.input = nn.Conv2d(img_channels, 64, 3, padding=1)
        self.down1 = Block(64, 128)
        self.down2 = Block(128, 256)
        self.bottleneck1 = Block(256, 512)
        self.bottleneck2 = Block(512, 512)
        self.bottleneck3 = Block(512, 256)
        self.up1 = Block(256, 128)
        self.up2 = Block(128, 64)
        self.output = nn.Conv2d(64, img_channels, 3, padding=1)

        self.embedding = nn.Embedding(1000, 128)

    def forward(self, x, t):
        t = self.embedding(t)
        x1 = self.input(x) # 64x64x64
        x2 = F.max_pool2d(x1, 2)
        x2 = self.down1(x2, t) # 32x32x128
        x3 = F.max_pool2d(x2, 2)
        x3 = self.down2(x3, t) # 16x16x256
        
        x = self.bottleneck1(x3, t)
        x = self.bottleneck2(x, t)
        x = self.bottleneck3(x, t) + x3 # 16x16x256

        x = F.interpolate(x, scale_factor=2) 
        x = self.up1(x, t) + x2 # 32x32x128
        x = F.interpolate(x, scale_factor=2)
        x = self.up2(x, t) + x1 # 64x64x64
        x = self.output(x) 
        return x