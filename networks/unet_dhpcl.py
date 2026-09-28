from __future__ import division, print_function

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LDE import LDE, patch_divide, patch_reverse

from .guided_diffusion.gaussian_diffusion import (
    get_named_beta_schedule, ModelMeanType, ModelVarType, LossType,
)
from .guided_diffusion.respace import SpacedDiffusion, space_timesteps
from .guided_diffusion.resample import UniformSampler

def get_timestep_embedding(timesteps, embedding_dim):

    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):

    return x * torch.sigmoid(x)

def kaiming_normal_init_weight(model):

    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):

    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, num_freqs=8):
        super().__init__()
        freqs = (2.0 ** torch.arange(num_freqs).float()) * math.pi
        self.register_buffer('freqs', freqs, persistent=False)

    def forward(self, gamma):

        x = gamma.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

class GammaModulator(nn.Module):

    def __init__(self, out_channels, num_freqs=8, hidden_dim=128):
        super().__init__()
        self.out_channels = out_channels
        self.pe = SinusoidalPositionalEncoding(num_freqs)
        self.linear1 = nn.Linear(2 * num_freqs, hidden_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, gamma, feat):

        if gamma.dim() == 4:
            gamma = gamma.squeeze(1)

        gamma = F.adaptive_avg_pool2d(gamma.unsqueeze(1), feat.shape[-2:]).squeeze(1)

        e = self.pe(gamma)
        e = self.linear2(self.act(self.linear1(e)))
        e = self.proj(e)
        e = e.permute(0, 3, 1, 2).contiguous()

        return feat + e

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),)
        self.conv1 = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

        self.temb_proj = torch.nn.Linear(512, out_channels)

    def forward(self, x, temb):

        x = self.conv0(x)
        x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        x = self.conv1(x)
        return x

class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x, temb):
        x = self.maxpool(x)
        x = self.conv(x, temb)
        return x

class UpBlock(nn.Module):

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2, temb):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, temb)

class Encoder(nn.Module):

    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)

        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x, temb, embeddings=None):
        x0 = self.in_conv(x, temb)
        if embeddings is not None:
            x0 = x0 + embeddings[0]
        x1 = self.down1(x0, temb)
        if embeddings is not None:
            x1 = x1 + embeddings[1]
        x2 = self.down2(x1, temb)
        if embeddings is not None:
            x2 = x2 + embeddings[2]
        x3 = self.down3(x2, temb)
        if embeddings is not None:
            x3 = x3 + embeddings[3]
        x4 = self.down4(x3, temb)
        if embeddings is not None:
            x4 = x4 + embeddings[4]
        return [x0, x1, x2, x3, x4]

class Decoder(nn.Module):

    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.bilinear = self.params['bilinear']
        self.out_chns = self.params['out_chns']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.out_chns, kernel_size=3, padding=1)

    def forward(self, feature, temb, out_multi=False):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        outs = []
        x = self.up1(x4, x3, temb)
        outs.append(x)
        x = self.up2(x, x2, temb)
        outs.append(x)
        x = self.up3(x, x1, temb)
        outs.append(x)
        x = self.up4(x, x0, temb)
        output = self.out_conv(x)
        outs.append(output)
        if out_multi:
            return outs
        return output

class DeUNet(nn.Module):

    def __init__(self, use_lde=True, lde_max_patch=8):
        super(DeUNet, self).__init__()
        self.ft_chns = [256, 384, 512]

        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], 0.0)
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], 0.0)

        self.up1 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128, 512),
            torch.nn.Linear(512, 512),
        ])

        self.use_lde = use_lde
        if use_lde:
            K = lde_max_patch

            self.lde_shallow = LDE(dim=self.ft_chns[0], qk_dim=32, mlp_dim=64,
                                   heads=4, max_patch_size=K)

            self.lde_mid = LDE(dim=self.ft_chns[1], qk_dim=48, mlp_dim=96,
                               heads=6, max_patch_size=K)

            self.lde_bottleneck = LDE(dim=self.ft_chns[2], qk_dim=64, mlp_dim=128,
                                      heads=8, max_patch_size=K)

    def forward(self, x, temb, embeddings=None):

        temb = get_timestep_embedding(temb, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        B, C, H, W = x.shape
        x0 = x
        if embeddings is not None and len(embeddings) == 1:

            x0 = x0 + embeddings[0]

        if self.use_lde:
            x0 = self.lde_shallow(x0)

        x1 = self.down1(x0, temb)
        if embeddings is not None and len(embeddings) > 1:
            x1 = x1 + embeddings[-2]

        if self.use_lde:
            x1 = self.lde_mid(x1)

        x2 = self.down2(x1, temb)
        if embeddings is not None and len(embeddings) > 1:
            x2 = x2 + embeddings[-1]

        if self.use_lde:
            x2 = self.lde_bottleneck(x2)

        x = self.up1(x2, x1, temb)
        x = self.up2(x, x0, temb)

        assert x.shape == (B, C, H, W)
        return x

class DiffUNet(nn.Module):

    def __init__(self, ts=1000, ts_sample=10, ldm_sch='linear',
                 use_lde=True, lde_max_patch=8) -> None:
        super().__init__()

        self.model = DeUNet(use_lde=use_lde, lde_max_patch=lde_max_patch)
        betas = get_named_beta_schedule(ldm_sch, ts)

        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(ts, [ts]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,)

        self.sample_diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(ts, [ts_sample]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,)

        self.sampler = UniformSampler(ts)

    def forward(self, x=None, pred_type=None, step=None, embeddings=None):

        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "q_sample_with_t":

            noise = torch.randn_like(x).to(x.device)
            return self.diffusion.q_sample(x, step, noise=noise), noise

        elif pred_type == "denoise":

            return self.model(x, temb=step, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            if embeddings is not None and len(embeddings) == 1:
                sample_out = self.sample_diffusion.ddim_sample_loop(
                    self.model, embeddings[0].shape,
                    model_kwargs={"embeddings": embeddings})
            else:
                sample_out = self.sample_diffusion.ddim_sample_loop(
                    self.model, embeddings[-3].shape,
                    model_kwargs={"embeddings": embeddings[-2:]})
            sample_out = sample_out["pred_xstart"]
            return sample_out

class UNet_LDMV2(nn.Module):

    def __init__(self, in_chns, class_num, out_chns, ldm_method='adaptor',
                 ldm_beta_sch='linear',
                 ts=1000, ts_sample=10,
                 use_lde=True, num_freqs=8, lde_max_patch=8):
        super(UNet_LDMV2, self).__init__()

        self.num_timesteps = ts

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'out_chns': out_chns,
                  'bilinear': False,
                  'acti_func': 'relu'}

        params2 = {'in_chns': in_chns - 3,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'out_chns': out_chns,
                   'bilinear': False,
                   'acti_func': 'relu'}

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128, 512),
            torch.nn.Linear(512, 512),
        ])

        self.encoder = Encoder(params)
        self.embedder = Encoder(params2)
        self.decoder = Decoder(params)

        self.deunet = DiffUNet(ts=ts, ts_sample=ts_sample,
                               ldm_sch=ldm_beta_sch, use_lde=use_lde,
                               lde_max_patch=lde_max_patch)
        self.de_loss = nn.MSELoss()

        self.gamma_modulator = GammaModulator(
            out_channels=params['feature_chns'][-1], num_freqs=num_freqs)

        self.ldm_method = ldm_method
        if ldm_method == 'adaptor':
            self.adaptor = ConvBlock(512, 256, 0.0)

    def get_lat_loss(self, pred, gt):

        return self.de_loss(pred, gt)

    def forward(self, x, t=None, image=None, training=True, good=None, gamma=None):

        B = x.shape[0]

        if t is None:
            t = torch.randint(0, self.num_timesteps, (B,), device=x.device)
        elif not torch.is_tensor(t):
            t = torch.full((B,), float(t), device=x.device)
        t = t.to(x.device)
        t_long = t.long()

        temb = get_timestep_embedding(t.float(), 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        if image is not None:
            img_embeddings = self.embedder(image, temb)
            x_in = torch.cat([image, x], dim=1)
        else:
            img_embeddings = None
            x_in = x

        feature = self.encoder(x_in, temb, img_embeddings)
        F_feat = feature[-1]

        if gamma is not None:
            F_prime = self.gamma_modulator(gamma, F_feat)
        else:

            F_prime = F_feat

        if training:

            if image is not None:
                target_in = torch.cat([image, good], dim=1)
            else:
                target_in = good
            x_start = self.encoder(target_in, temb, img_embeddings)[-1].detach()

            x_t, noise = self.deunet(x=x_start, pred_type="q_sample_with_t", step=t_long)

            pred_xstart = self.deunet(x=x_t, step=t_long, pred_type="denoise",
                                      embeddings=[F_prime])

            if self.ldm_method == 'adaptor':
                feat_ref = torch.cat([F_feat, pred_xstart], dim=1)
            elif self.ldm_method == 'add':
                feat_ref = F_feat + pred_xstart
            else:
                feat_ref = pred_xstart

            lat_loss = self.get_lat_loss(pred_xstart, x_start)

        else:
            assert good is None

            sample_xstart = self.deunet(pred_type="ddim_sample", embeddings=[F_prime])

            if self.ldm_method == 'adaptor':
                feat_ref = torch.cat([F_feat, sample_xstart], dim=1)
            elif self.ldm_method == 'add':
                feat_ref = F_feat + sample_xstart
            else:
                feat_ref = sample_xstart

        if self.ldm_method == 'adaptor':
            feature[-1] = self.adaptor(feat_ref, temb)
        else:
            feature[-1] = feat_ref

        output = self.decoder(feature, temb, out_multi=True)
        output = output[-1]

        if training:
            return lat_loss, output
        else:
            return output

if __name__ == '__main__':
    model = UNet_LDMV2(4, 2, 2, ts=20, ts_sample=4)
    x = torch.rand(1, 3, 256, 256)
    image = torch.rand(1, 1, 256, 256)
    gamma = torch.rand(1, 256, 256)
    lat_loss, out = model(x, None, image, True, good=x, gamma=gamma)
    print(lat_loss, out.shape)
    out = model(x, None, image, False)
    print(out.shape)
