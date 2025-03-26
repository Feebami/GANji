import torch
import torch.nn as nn
import torch.nn.functional as F

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