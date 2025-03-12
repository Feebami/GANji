import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(True),
        )

        self.emb_layer = nn.Linear(emb_dim, out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t):
        return self.block(x) + self.shortcut(x) + self.emb_layer(t).unsqueeze(-1).unsqueeze(-1)
    
class AttentionGate(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.ReLU(True),
        )
    def forward(self, x):
        x = torch.cat([torch.max(x, 1, keepdim=True)[0], torch.mean(x, 1, keepdim=True)], dim=1)
        x = self.conv(x)
        scale = F.sigmoid(x)
        return scale * x
    
class TripletAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.hw = AttentionGate()
    def forward(self, x):
        squash_h = x.permute(0, 2, 1, 3)
        squash_w = x.permute(0, 3, 2, 1)
        cw_atten = self.cw(squash_h).permute(0, 2, 1, 3)
        hc_atten = self.hc(squash_w).permute(0, 3, 2, 1)
        hw_atten = self.hw(x)
        return (cw_atten + hc_atten + hw_atten) / 3
    
class UNet(nn.Module):
    def __init__(self, img_channels, n_steps):
        super().__init__()
        self.input = nn.Conv2d(img_channels, 64, 3, padding=1)

        self.down1 = Block(64, 128)
        self.down2 = Block(128, 256)
        self.down3 = Block(256, 256)

        self.bottleneck1 = Block(256, 512)
        self.bottleneck2 = Block(512, 512)
        self.bottleneck3 = Block(512, 256)

        # self.attn1 = TripletAttention()
        self.up1 = Block(512, 128)
        # self.attn2 = TripletAttention()
        self.up2 = Block(256, 64)
        # self.attn3 = TripletAttention()
        self.up3 = Block(128, 64)

        self.output = nn.Conv2d(64, img_channels, 3, padding=1)

        self.embedding = self.sinusoidal_embeddings(n_steps).to(device)

    def forward(self, x, t):
        t = self.embedding[t]
        x1 = self.input(x) # 64x64x64
        x2 = F.max_pool2d(x1, 2)
        x2 = self.down1(x2, t) # 128x32x32
        x3 = F.max_pool2d(x2, 2)
        x3 = self.down2(x3, t) # 256x16x16
        x4 = F.max_pool2d(x3, 2)
        x4 = self.down3(x4, t) # 256x8x8
        
        x = self.bottleneck1(x4, t) # 512x8x8
        x = self.bottleneck2(x, t) # 512x8x8
        x = self.bottleneck3(x, t) # 256x8x8

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) # 256x16x16
        # x3 = self.attn1(x3)
        x = torch.cat([x, x3], dim=1) # 512x16x16
        x = self.up1(x, t) # 128x16x16
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) # 128x32x32
        # x2 = self.attn2(x2)
        x = torch.cat([x, x2], dim=1) # 256x32x32
        x = self.up2(x, t) # 64x32x32
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) # 64x64x64
        # x1 = self.attn3(x1)
        x = torch.cat([x, x1], dim=1) # 128x64x64
        x = self.up3(x, t) # 64x64x64
        x = self.output(x) # img_channelsx64x64

        return x
    
    def sinusoidal_embeddings(self, t, emb_dim=256):
        denom = 10000 ** (torch.arange(0, emb_dim, 2).float() / emb_dim)
        positions = torch.arange(0, t).float().unsqueeze(1)
        embeddings = torch.zeros(t, emb_dim)
        embeddings[:, 0::2] = torch.sin(positions / denom)
        embeddings[:, 1::2] = torch.cos(positions / denom)
        return embeddings