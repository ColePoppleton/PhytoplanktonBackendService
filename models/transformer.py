import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer
from typing import Tuple, Dict


class SpatiotemporalLSTM(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=kernel_size // 2
        )
        self.hidden_dim = hidden_dim

    def forward(self, x, state: Tuple[torch.Tensor, torch.Tensor]):
        h_cur, c_cur = state
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)

        c_next = torch.sigmoid(f) * c_cur + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, (h_next, c_next)


class SwinPredictor(nn.Module):
    def __init__(self, img_size: int = 64):
        super().__init__()
        self.swin = SwinTransformer(
            img_size=img_size,
            patch_size=4,
            in_chans=1,
            num_classes=0,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        self.head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_size * img_size)
        )

    def forward(self, x):
        batch, seq, c, h, w = x.shape
        x = x.view(batch * seq, c, h, w)
        features = self.swin(x)
        out = self.head(features)
        return out.view(batch, seq, h, w)[:, -1, :, :].unsqueeze(1)