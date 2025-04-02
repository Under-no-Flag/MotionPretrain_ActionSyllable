


import torch
from torch import nn
from model.blocks.attention import CausalSelfAttention

from model.MotionGPT import SpatioTemporalAttention
from model.blocks.attention import Base_STAttention
class STBlock(nn.Module):
    def __init__(self, config, in_factor=None, out_factor=None):
        super().__init__()
        in_dim = config.n_embd * (in_factor if in_factor is not None else 1)
        out_dim = config.n_embd * (out_factor if out_factor is not None else 1)
        self.ln1 = nn.LayerNorm(in_dim)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = Base_STAttention(dim=in_dim,n_heads=config.n_head)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, out_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False, valid_mask=None, in_residual=True, out_residual=True):

        if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn= self.attn(self.ln1(x))
        x = x + attn if in_residual else attn
        x = x + self.mlp(self.ln2(x)) if out_residual else self.mlp(self.ln2(x))

        return x


class Block(nn.Module):
    def __init__(self, config, in_factor=None, out_factor=None):
        super().__init__()
        in_dim = config.n_embd * (in_factor if in_factor is not None else 1)
        out_dim = config.n_embd * (out_factor if out_factor is not None else 1)
        self.ln1 = nn.LayerNorm(in_dim)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, in_dim=in_dim if in_factor is not None else None)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, out_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False, valid_mask=None, in_residual=True, out_residual=True):

        if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), layer_past=layer_past,
                                  valid_mask=valid_mask)
        x = x + attn if in_residual else attn
        x = x + self.mlp(self.ln2(x)) if out_residual else self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x