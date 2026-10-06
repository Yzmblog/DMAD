"""Pretrained-feature critic (Projected GAN / D2O) with two head sets, used by --pretrained_critic.

Adapted from D2O (https://github.com/Zyriix/D2O, pg_modules/ and feature_networks/), which follows Projected GAN
(Sauer et al.): frozen ImageNet-pretrained feature networks (vgg16_bn from torchvision, tf_efficientnet_lite0 from
timm), frozen random cross-channel/cross-scale projections, spectrally normalized multi-scale conv heads with
per-pixel logits and projection class conditioning. Differences from D2O: no DiffAugment; two independent head sets
(real: T vs G, teacher: Q vs G) on one shared feature forward; the class embedding is initialized from the
tf_efficientnet_lite0 classifier weight; a plain-PyTorch fully connected layer replaces StyleGAN2's.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# --------------------------------------------------------------------------- #
# blocks (D2O pg_modules/blocks.py)
# --------------------------------------------------------------------------- #

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def NormLayer(c, mode='batch'):
    if mode == 'group':
        return nn.GroupNorm(c // 2, c)
    elif mode == 'batch':
        return nn.BatchNorm2d(c)


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, width=1):
        super().__init__()
        self.main = nn.Sequential(
            conv2d(in_planes, out_planes * width, 4, 2, 1, bias=True),
            NormLayer(out_planes * width),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, expand=False, lowest=False):
        super().__init__()
        out_features = features // 2 if expand else features
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1,
                                  padding=0, bias=True, groups=1)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output = output + xs[1]
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return self.out_conv(output)


# --------------------------------------------------------------------------- #
# pretrained feature networks (D2O feature_networks/pretrained_builder.py)
# --------------------------------------------------------------------------- #

# D2O normstats (feature_networks/constants.py groups): vgg16_bn is TORCHVISION ->
# NORMALIZED_IMAGENET; tf_efficientnet_lite0 is EFFNETS_INCEPTION -> NORMALIZED_INCEPTION.
NORMSTATS = {
    'vgg16_bn': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    'tf_efficientnet_lite0': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
}


def _feature_splitter(model, idcs):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.features[:idcs[0]])
    pretrained.layer1 = nn.Sequential(model.features[idcs[0]:idcs[1]])
    pretrained.layer2 = nn.Sequential(model.features[idcs[1]:idcs[2]])
    pretrained.layer3 = nn.Sequential(model.features[idcs[2]:idcs[3]])
    return pretrained


def _make_efficientnet(model):
    pretrained = nn.Module()
    # timm >=0.6 fuses bn1+act1 into BatchNormAct2d; D2O's separate act1 exists only in old timm
    act1 = getattr(model, 'act1', None) or nn.Identity()
    pretrained.layer0 = nn.Sequential(
        model.conv_stem, model.bn1, act1, *model.blocks[0:2])
    pretrained.layer1 = nn.Sequential(*model.blocks[2:3])
    pretrained.layer2 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer3 = nn.Sequential(*model.blocks[5:9])
    return pretrained


def calc_channels(pretrained, inp_res=224):
    channels = []
    tmp = torch.zeros(1, 3, inp_res, inp_res)
    with torch.no_grad():
        tmp = pretrained.layer0(tmp); channels.append(tmp.shape[1])
        tmp = pretrained.layer1(tmp); channels.append(tmp.shape[1])
        tmp = pretrained.layer2(tmp); channels.append(tmp.shape[1])
        tmp = pretrained.layer3(tmp); channels.append(tmp.shape[1])
    return channels


def _make_pretrained(backbone):
    """vgg16_bn (torchvision, ImageNet1k) or tf_efficientnet_lite0 (timm, ImageNet1k).
    Returns (feature_module, lite0_classifier_weight_or_None)."""
    embed_weight = None
    if backbone == 'vgg16_bn':
        import torchvision.models as zoomodels
        model = zoomodels.vgg16_bn(weights=zoomodels.VGG16_BN_Weights.IMAGENET1K_V1)
        pretrained = _feature_splitter(model, [13, 23, 33, 43])   # D2O idcs
    elif backbone == 'tf_efficientnet_lite0':
        import timm
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_efficientnet(model)
        embed_weight = model.classifier.weight.detach().clone()   # [1000, 1280]
    else:
        raise NotImplementedError(f"pg_critic backbone {backbone} (supported: vgg16_bn, "
                                  f"tf_efficientnet_lite0)")
    pretrained.CHANNELS = calc_channels(pretrained)
    return pretrained, embed_weight


# --------------------------------------------------------------------------- #
# projector: CCM + CSM random projections (D2O pg_modules/projector.py, proj_type=2)
# --------------------------------------------------------------------------- #

def _make_scratch_ccm(scratch, in_channels, cout, expand=False):
    out_channels = [cout, cout * 2, cout * 4, cout * 8] if expand else [cout] * 4
    scratch.layer0_ccm = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer1_ccm = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer2_ccm = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer3_ccm = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.CHANNELS = out_channels
    return scratch


def _make_scratch_csm(scratch, in_channels, cout, expand):
    scratch.layer3_csm = FeatureFusionBlock(in_channels[3], expand=expand, lowest=True)
    scratch.layer2_csm = FeatureFusionBlock(in_channels[2], expand=expand)
    scratch.layer1_csm = FeatureFusionBlock(in_channels[1], expand=expand)
    scratch.layer0_csm = FeatureFusionBlock(in_channels[0])
    # last refinenet does not expand to save channels in higher dimensions
    scratch.CHANNELS = [cout, cout, cout * 2, cout * 4] if expand else [cout] * 4
    return scratch


class F_RandomProj(nn.Module):
    """Frozen pretrained backbone + random CCM/CSM projections (PG proj_type=2)."""

    def __init__(self, backbone="vgg16_bn", im_res=224, cout=64, expand=True):
        super().__init__()
        self.backbone = backbone
        self.normstats = NORMSTATS[backbone]
        self.pretrained, self.embed_weight = _make_pretrained(backbone)
        self.RESOLUTIONS = [im_res // 4, im_res // 8, im_res // 16, im_res // 32]
        scratch = nn.Module()
        scratch = _make_scratch_ccm(scratch, in_channels=self.pretrained.CHANNELS,
                                    cout=cout, expand=expand)
        scratch = _make_scratch_csm(scratch, in_channels=scratch.CHANNELS,
                                    cout=cout, expand=expand)
        self.scratch = scratch
        # CSM upsamples x2 so the feature map resolution doubles
        self.RESOLUTIONS = [res * 2 for res in self.RESOLUTIONS]
        self.CHANNELS = scratch.CHANNELS

    def forward(self, x):
        out0 = self.pretrained.layer0(x)
        out1 = self.pretrained.layer1(out0)
        out2 = self.pretrained.layer2(out1)
        out3 = self.pretrained.layer3(out2)
        out0 = self.scratch.layer0_ccm(out0)
        out1 = self.scratch.layer1_ccm(out1)
        out2 = self.scratch.layer2_ccm(out2)
        out3 = self.scratch.layer3_ccm(out3)
        out3_sm = self.scratch.layer3_csm(out3)
        out2_sm = self.scratch.layer2_csm(out3_sm, out2)
        out1_sm = self.scratch.layer1_csm(out2_sm, out1)
        out0_sm = self.scratch.layer0_csm(out1_sm, out0)
        return {'0': out0_sm, '1': out1_sm, '2': out2_sm, '3': out3_sm}


# --------------------------------------------------------------------------- #
# discriminator heads (D2O pg_modules/discriminator.py)
# --------------------------------------------------------------------------- #

class FullyConnectedLayer(nn.Module):
    """Plain-torch equivalent of the StyleGAN2 FC layer D2O uses for embed_proj
    (equalized-lr init, lrelu with sqrt(2) gain)."""

    def __init__(self, in_features, out_features, activation='linear', lr_multiplier=1.0):
        super().__init__()
        self.activation = activation
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_multiplier)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.weight_gain = lr_multiplier / math.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        out = F.linear(x, self.weight * self.weight_gain, self.bias * self.bias_gain)
        if self.activation == 'lrelu':
            out = F.leaky_relu(out, 0.2) * math.sqrt(2)
        return out


# midas channel table (D2O SingleDisc)
NFC_MIDAS = {1: 2048, 2: 1024, 4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
             256: 32, 512: 16, 1024: 8}


def _snap_start_sz(start_sz):
    if start_sz not in NFC_MIDAS:
        sizes = np.array(list(NFC_MIDAS.keys()))
        start_sz = int(sizes[np.argmin(abs(sizes - start_sz))])
    return start_sz


class SingleDiscCond(nn.Module):
    """One per-scale conv head with projection conditioning (D2O SingleDiscCond; per-pixel
    logits at end_sz)."""

    def __init__(self, nc, start_sz, end_sz=8, c_dim=1000, cmap_dim=64, embed_weight=None):
        super().__init__()
        self.cmap_dim = cmap_dim
        start_sz = _snap_start_sz(start_sz)
        nfc = dict(NFC_MIDAS)
        nfc[start_sz] = nc
        layers = []
        while start_sz > end_sz:
            layers.append(DownBlock(nfc[start_sz], nfc[start_sz // 2]))
            start_sz = start_sz // 2
        self.main = nn.Sequential(*layers)
        kernel = 4 if end_sz >= 8 else max(1, end_sz // 2)
        self.cls = conv2d(nfc[end_sz], self.cmap_dim, kernel, 1, 0, bias=False)
        # D2O loads pretrained ImageNet embeddings (efficientnet-lite0 classifier); build the
        # same thing from the timm classifier weight, random init as their rand_embedding fallback.
        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=False)
        else:
            self.embed = nn.Embedding(c_dim, 320)
        self.embed_proj = FullyConnectedLayer(self.embed.embedding_dim, self.cmap_dim,
                                              activation='lrelu')

    def forward(self, x, c):
        h = self.main(x)
        out = self.cls(h)                                       # [N, cmap_dim, h', w']
        cmap = self.embed_proj(self.embed(c.argmax(1))).unsqueeze(-1).unsqueeze(-1)
        out = (out * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        return out                                              # [N, 1, h', w']


class MultiScaleD(nn.Module):
    def __init__(self, channels, resolutions, num_discs=4, c_dim=1000, embed_weight=None):
        super().__init__()
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        mini_discs = []
        end_sz = 8 if self.disc_in_res[-1] >= 8 else self.disc_in_res[-1]
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            mini_discs += [str(i), SingleDiscCond(nc=cin, start_sz=res, end_sz=end_sz,
                                                  c_dim=c_dim, embed_weight=embed_weight)],
        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features, c):
        all_logits = []
        for k, disc in self.mini_discs.items():
            all_logits.append(disc(features[k], c).view(features[k].size(0), -1))
        return torch.cat(all_logits, dim=1)                     # [N, sum(spatial)]


# --------------------------------------------------------------------------- #
# top level: two-head pretrained-feature critic
# --------------------------------------------------------------------------- #

class PGTwoHeadCritic(nn.Module):
    """Frozen pretrained feature networks + two independent MultiScaleD head sets.

    forward(x, label_onehot) -> (d_real [N, K], d_teacher [N, K]) where K concatenates every
    backbone's per-scale per-pixel logits. Consumers reduce elementwise (the [N, S] multiscale
    convention), which reproduces D2O's per-pixel BCE as the average over positions.
    Input x is in [-1, 1] at any resolution; it is renormalized per backbone and bilinearly
    resized to 224 exactly as in D2O's ProjectedDiscriminator (interp224). Sigma is NOT an
    input: the D2O discriminator has no noise-level conditioning.
    """

    def __init__(self, backbones=("vgg16_bn", "tf_efficientnet_lite0"), c_dim=1000):
        super().__init__()
        feature_networks, discs_real, discs_teacher = [], [], []
        for bb_name in backbones:
            feat = F_RandomProj(bb_name)
            feature_networks.append([bb_name, feat])
        self.feature_networks = nn.ModuleDict(feature_networks)
        # lite0 classifier weight = D2O's pretrained ImageNet embedding; shared source for
        # every head's embed (each head clones its own trainable copy, as in D2O where each
        # SingleDiscCond loads the pickle independently).
        embed_weight = None
        for _, feat in self.feature_networks.items():
            if feat.embed_weight is not None:
                embed_weight = feat.embed_weight
        for bb_name, feat in self.feature_networks.items():
            discs_real.append([bb_name, MultiScaleD(feat.CHANNELS, feat.RESOLUTIONS,
                                                    c_dim=c_dim, embed_weight=embed_weight)])
            discs_teacher.append([bb_name, MultiScaleD(feat.CHANNELS, feat.RESOLUTIONS,
                                                       c_dim=c_dim, embed_weight=embed_weight)])
        self.discs_real = nn.ModuleDict(discs_real)
        self.discs_teacher = nn.ModuleDict(discs_teacher)
        # everything upstream of the heads (pretrained networks and random projections) is frozen
        self.feature_networks.requires_grad_(False)

    def train(self, mode=True):
        # the frozen feature networks (incl. their BatchNorms) always run in eval mode
        super().train(mode)
        self.feature_networks.train(False)
        return self

    def _features(self, x):
        feats = {}
        for bb_name, feat in self.feature_networks.items():
            x_n = x.add(1).div(2)                               # [-1,1] -> [0,1]
            mean = x_n.new_tensor(feat.normstats['mean']).view(1, 3, 1, 1)
            std = x_n.new_tensor(feat.normstats['std']).view(1, 3, 1, 1)
            x_n = (x_n - mean) / std
            x_n = F.interpolate(x_n, 224, mode='bilinear', align_corners=False)
            feats[bb_name] = feat(x_n)
        return feats

    def forward(self, x, c):
        feats = self._features(x)
        d_real, d_teacher = [], []
        for bb_name in self.feature_networks.keys():
            d_real.append(self.discs_real[bb_name](feats[bb_name], c))
            d_teacher.append(self.discs_teacher[bb_name](feats[bb_name], c))
        return torch.cat(d_real, dim=1), torch.cat(d_teacher, dim=1)
