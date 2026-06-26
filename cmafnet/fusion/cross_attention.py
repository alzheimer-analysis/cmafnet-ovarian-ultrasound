import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def _split(self, x):
        b, n, d = x.shape
        x = x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        return x

    def forward(self, query_tokens, kv_token):
        q = self.norm_q(query_tokens)
        kv = self.norm_kv(kv_token)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        q = self._split(self.q_proj(q))
        k = self._split(k.unsqueeze(1))
        v = self._split(v.unsqueeze(1))
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(query_tokens.shape)
        return query_tokens + self.out_proj(out)


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.img_to_clin = CrossModalBlock(dim)
        self.clin_to_img = CrossModalBlock(dim)

    def forward(self, img_tokens, clin_vec):
        b, c, h, w = img_tokens.shape
        seq = img_tokens.flatten(2).transpose(1, 2)
        clin_tok = clin_vec.unsqueeze(1)
        seq = self.clin_to_img(seq, clin_vec)
        clin_tok = self.img_to_clin(clin_tok, seq.mean(dim=1))
        fused_map = seq.transpose(1, 2).reshape(b, c, h, w)
        return fused_map, clin_tok.squeeze(1)


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def forward(self, img_vec, clin_vec):
        z = torch.cat([img_vec, clin_vec], dim=-1)
        g = self.gate(z)
        m = self.mix(z)
        return g * img_vec + (1.0 - g) * clin_vec + m
