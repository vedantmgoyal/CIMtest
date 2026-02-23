import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat


# === 保留原版 SpectralFormer 组件 ===
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))
        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](
                        torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask=mask)
                x = ff(x)
                nl += 1
        return x


class ViT(nn.Module):
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 channels=1, dim_head=16, dropout=0., emb_dropout=0., mode='ViT'):
        super().__init__()
        patch_dim = image_size ** 2 * near_band
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask=None):
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, mask)
        latent = self.to_latent(x[:, 0])
        return self.mlp_head(latent), latent


# ==============================================================================
# 【核心升级】 空间因果干预模块 (Spatial Causal Intervention Module)
# ==============================================================================
class SpatialCausalInterventionModule(nn.Module):
    def __init__(self, patch_size, band, hidden_dim=64):
        super().__init__()
        self.num_pixels = patch_size * patch_size  # 空间像素总数 (例如 7x7=49)

        # 提取整条光谱曲线的特征进行打分
        self.proj = nn.Linear(band, hidden_dim)
        mask_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.mask_transformer = nn.TransformerEncoder(mask_layer, num_layers=1)
        self.mask_mlp = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x, x_hetero=None):
        # x 形状: [Batch, 波段数(200), 空间展平(例如147)]
        b, c, p = x.shape
        band_patches = p // self.num_pixels  # 比如 147 // 49 = 3
        nn_idx = band_patches // 2  # 取中间的那个组

        # 【关键1】提取中心波段的 49 个空间像素: [Batch, 49, 200]
        center_spatial = x[:, :, nn_idx * self.num_pixels: (nn_idx + 1) * self.num_pixels].permute(0, 2, 1)

        feat = self.proj(center_spatial)
        mask_logits = self.mask_mlp(self.mask_transformer(feat))

        # 对 49 个空间位置打分
        M = F.gumbel_softmax(mask_logits, tau=1.0, hard=True)
        causal_mask = M[:, :, 1].unsqueeze(-1)  # 形状: [Batch, 49, 1]

        # 【关键2】将这个空间 Mask 复制，应用到所有 band_patch 上: [Batch, 147, 1]
        causal_mask_full = causal_mask.repeat(1, band_patches, 1)

        # 旋转维度以符合 x 的形状: [Batch, 1, 147]
        causal_mask_full = causal_mask_full.permute(0, 2, 1)

        # 执行替换！由于 mask 在维度1（波段维度）上是广播的，这保证了整条光谱曲线是一起被保留或替换的！
        if x_hetero is not None:
            counterfactual_x = causal_mask_full * x + (1 - causal_mask_full) * x_hetero
        else:
            counterfactual_x = causal_mask_full * x

        return counterfactual_x


class CausalSpectralFormer(nn.Module):
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, mode='ViT'):
        super().__init__()
        # 挂载空间因果干预模块
        self.causal_intervention = SpatialCausalInterventionModule(patch_size=image_size, band=num_patches,
                                                                   hidden_dim=dim)
        self.spectral_former = ViT(image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim,
                                   mode=mode)

    def forward(self, x, x_hetero=None):
        # 1. 直接在输入的最早期进行“空间因果清洗”
        if self.training and x_hetero is not None:
            x_cf = self.causal_intervention(x, x_hetero)

            # 2. 清洗完后，送入 SpectralFormer 主体
            logits_org, z_org = self.spectral_former(x)
            logits_cf, z_cf = self.spectral_former(x_cf)
            return logits_org, logits_cf, z_org, z_cf
        else:
            x_cf = self.causal_intervention(x, None)
            logits_cf, _ = self.spectral_former(x_cf)
            return logits_cf