"""
unet3plus.py

UNet3+ (Huang et al. 2020) — Full-Scale Connected UNet.
https://arxiv.org/abs/2004.08790

Key difference from UNet / UNet++:
    Each decoder node receives feature maps from ALL encoder scales
    AND ALL other decoder scales simultaneously, not just the
    corresponding encoder scale (UNet) or progressively refined
    intermediate nodes (UNet++).

    For a 5-scale encoder, each decoder node receives 9 inputs
    (5 encoder + 4 decoder), each projected to D channels, then fused:
        9 × D → 3×3 Conv BN ReLU → D channels

Interface:
    Matches segmentation_models_pytorch conventions so it is
    drop-in compatible with train.py build_model():
        encoder_name, encoder_weights, in_channels, classes, activation

Training mode:
    Single output head — no deep supervision.
    Deep supervision is deferred to a post-sweep ablation (FINETUNE scope).
    This ensures identical training conditions across all Wave 1 architectures.

Encoders tested (all pass, output 256×256×classes):
    resnet34     26.6M params
    resnet50     37.1M params
    densenet121  17.6M params
    densenet169  25.8M params
    efficientnet-b3  14.9M params
    efficientnet-b4  22.1M params
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# ─── Building block ───────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─── UNet3+ ───────────────────────────────────────────────────────────────────

class UNet3Plus(nn.Module):
    """
    UNet3+ with full-scale skip connections.

    Architecture:
        Encoder  : any smp-compatible encoder (resnet, densenet, efficientnet)
        Decoder  : 5 decoder nodes, each receiving from all encoder scales
                   and all other decoder scales (full-scale connectivity)
        Head     : 1×1 Conv → classes (no activation — raw logits)

    Args:
        encoder_name     : smp encoder name (e.g. 'densenet169')
        encoder_weights  : 'imagenet' or None
        in_channels      : number of input image channels (1 for grayscale)
        classes          : number of output segmentation classes (3 for BGW)
        activation       : ignored (kept for smp interface compatibility)
        decoder_channels : feature channels D at each decoder node (default 64)
    """

    def __init__(
        self,
        encoder_name:     str   = 'resnet34',
        encoder_weights:  str   = 'imagenet',
        in_channels:      int   = 1,
        classes:          int   = 3,
        activation               = None,
        decoder_channels: int   = 64,
    ):
        super().__init__()

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )
        # out_channels[0] is the raw input passthrough — skip it
        enc_ch = self.encoder.out_channels[1:]   # (ch_s1, ch_s2, ch_s3, ch_s4, ch_s5)
        n      = len(enc_ch)                     # 5 encoder scales
        D      = decoder_channels
        self._n = n
        self._D = D

        # ── Projection convs ──────────────────────────────────────────────────
        # Each decoder node d receives:
        #   n encoder inputs  — enc_proj[d][e] projects encoder scale e → D
        #   n-1 decoder inputs — dec_proj[d][k] projects other decoder scale → D
        #   total inputs = 2n-1 = 9  (for n=5)
        #   fused: (2n-1)×D channels → 3×3 ConvBnRelu → D channels

        n_inputs = 2 * n - 1
        fuse_ch  = n_inputs * D

        self.enc_proj = nn.ModuleList([
            nn.ModuleList([ConvBnRelu(enc_ch[e], D) for e in range(n)])
            for d in range(n)
        ])

        self.dec_proj = nn.ModuleList([
            nn.ModuleList([ConvBnRelu(D, D) for _ in range(n - 1)])
            for d in range(n)
        ])

        self.fuse = nn.ModuleList([
            ConvBnRelu(fuse_ch, D) for _ in range(n)
        ])

        # ── Classification head ───────────────────────────────────────────────
        self.head = nn.Conv2d(D, classes, kernel_size=1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (N, C, H, W) input tensor

        Returns:
            logits : (N, classes, H, W) — same spatial size as input
        """
        features = self.encoder(x)
        enc = list(features[1:])    # 5 encoder feature maps, high→low resolution
                                    # enc[0]: 128×128, enc[4]: 8×8  (for 256 input)

        n   = self._n
        D   = self._D
        dec = [None] * n

        # Decode deepest-first (largest index = smallest spatial = semantically richest)
        for d in reversed(range(n)):
            target_size = enc[d].shape[2:]   # (H, W) at this decoder scale
            parts       = []

            # ── Encoder inputs ─────────────────────────────────────────────────
            for e in range(n):
                feat = self.enc_proj[d][e](enc[e])
                feat = F.interpolate(feat, size=target_size,
                                     mode='bilinear', align_corners=False)
                parts.append(feat)

            # ── Decoder inputs (all other decoder scales) ──────────────────────
            k = 0
            for other in range(n):
                if other == d:
                    continue
                if dec[other] is not None:
                    feat = self.dec_proj[d][k](dec[other])
                else:
                    # Node not yet computed — zero placeholder
                    feat = torch.zeros(
                        enc[d].shape[0], D, *target_size,
                        device=enc[d].device, dtype=enc[d].dtype,
                    )
                feat = F.interpolate(feat, size=target_size,
                                     mode='bilinear', align_corners=False)
                parts.append(feat)
                k += 1

            dec[d] = self.fuse[d](torch.cat(parts, dim=1))

        # Upsample shallowest decoder output (dec[0]) to input resolution
        out = F.interpolate(dec[0], size=x.shape[2:],
                            mode='bilinear', align_corners=False)
        return self.head(out)
