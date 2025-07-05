import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.norm = nn.GroupNorm(8, channels)
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, C, -1).permute(0, 2, 1)
        k = k.view(B, C, -1).permute(0, 2, 1)
        v = v.view(B, C, -1).permute(0, 2, 1)

        attn, _ = self.mha(q, k, v)
        out = attn.permute(0, 2, 1).view(B, C, H, W)
        return x + self.out(out)