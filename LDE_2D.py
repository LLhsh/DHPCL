import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def patch_divide(x, step, patch_size):

    b, c, h, w = x.shape
    ps = patch_size

    if h < ps or w < ps:
        ps = min(h, w)
        print(f"Warning: Input feature map size ({h}x{w}) is smaller than "
              f"patch size ({patch_size}). Adjusting patch size to {ps}.")

    nh = max(1, (h - ps) // step + 1)
    nw = max(1, (w - ps) // step + 1)

    crop_x = []
    for i in range(nh):
        for j in range(nw):

            start_h = min(i * step, h - ps)
            start_w = min(j * step, w - ps)
            crop_x.append(x[:, :, start_h:start_h + ps, start_w:start_w + ps])

    crop_x = torch.stack(crop_x, dim=1)
    return crop_x, nh, nw

def patch_reverse(crop_x, x, step, ps):

    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    count = torch.zeros_like(x)

    nh = max(1, (h - ps) // step + 1)
    nw = max(1, (w - ps) // step + 1)

    index = 0
    for i in range(nh):
        start_h = min(i * step, h - ps)
        for j in range(nw):
            start_w = min(j * step, w - ps)
            output[:, :, start_h:start_h + ps, start_w:start_w + ps] += crop_x[:, index]
            count[:, :, start_h:start_h + ps, start_w:start_w + ps] += 1.0
            index += 1

    output = output / count.clamp(min=1.0)
    return output

class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class dwconv(nn.Module):

    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features),
            nn.GELU()
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):

        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features,
                                   x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)

        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

class Attention(nn.Module):

    def __init__(self, dim, heads, qk_dim):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = (qk_dim // heads) ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
                      (q, k, v))

        out = F.scaled_dot_product_attention(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)

class LDE(nn.Module):

    def __init__(self, dim, qk_dim=36, mlp_dim=96, heads=4,
                 min_patch_size=4, max_patch_size=16):
        super().__init__()

        self.layer = nn.ModuleList([
            PreNorm(dim, Attention(dim, heads, qk_dim)),
            PreNorm(dim, ConvFFN(dim, mlp_dim)),
        ])

        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size

    def forward(self, x):
        _, _, h, w = x.shape
        min_hw = min(h, w)

        if min_hw <= self.max_patch_size:
            ps = min_hw
        else:
            ps = self.max_patch_size
        ps = max(self.min_patch_size, ps) if min_hw > self.min_patch_size else ps
        ps = int(max(1, min(ps, h, w)))

        step = max(1, ps // 2)

        attn, ff = self.layer

        if min_hw <= 2:
            x_flat = x.flatten(2).transpose(1, 2)
            x_flat = attn(x_flat) + x_flat
            x_flat = ff(x_flat, x_size=(h, w)) + x_flat
            x = x_flat.transpose(1, 2).reshape(x.shape)
            return x

        crop_x, nh, nw = patch_divide(x, step, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')

        crop_x = attn(crop_x) + crop_x

        crop_x = rearrange(crop_x, '(b n) (h w) c -> b n c h w', n=n, w=pw)

        x = patch_reverse(crop_x, x, step, ps)

        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)

        return x
