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
        self.down3 = Block(256, 256)
        self.bottleneck1 = Block(256, 512)
        self.bottleneck2 = Block(512, 512)
        self.bottleneck3 = Block(512, 256)
        self.up1 = Block(512, 128)
        self.up2 = Block(256, 64)
        self.up3 = Block(128, 64)
        self.output = nn.Conv2d(64, img_channels, 3, padding=1)

        self.embedding = nn.Embedding(1000, 128)

    def forward(self, x, t):
        t = self.embedding(t)
        x1 = self.input(x) # 64x64x64
        x2 = F.max_pool2d(x1, 2)
        x2 = self.down1(x2, t) # 128x32x32
        x3 = F.max_pool2d(x2, 2)
        x3 = self.down2(x3, t) # 256x16x16
        x4 = F.max_pool2d(x3, 2)
        x4 = self.down3(x4, t) # 256x8x8
        
        x = self.bottleneck1(x4, t)
        x = self.bottleneck2(x, t)
        x = self.bottleneck3(x, t)

        skip_scale = 1.0
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) # 256x16x16
        x = torch.cat([x, x3 * skip_scale], dim=1) # 512x16x16
        x = self.up1(x, t) # 128x16x16
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) # 128x32x32
        x = torch.cat([x, x2 * skip_scale], dim=1) # 256x32x32
        x = self.up2(x, t) # 64x32x32
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) # 64x64x64
        x = torch.cat([x, x1 * skip_scale], dim=1) # 128x64x64
        x = self.up3(x, t) # 64x64x64
        x = self.output(x)

        return x