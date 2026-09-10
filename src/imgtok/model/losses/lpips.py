import torch
import torch.nn as nn
from torchvision import models as tv_models

from typing import Optional


class ScalingLayer(nn.Module):
    """Normalize input using ImageNet statistics (like taming's implementation)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single 1x1 conv layer used in LPIPS for each VGG feature level."""

    def __init__(self, in_channels: int, out_channels: int = 1, use_dropout: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class VGG16Features(nn.Module):
    """Extract features from 5 layers of a frozen VGG-16 network."""

    LAYER_NAMES = ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
    CHANNELS = [64, 128, 256, 512, 512]

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        self.slices = nn.ModuleList()
        self._build_slices(vgg)
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def _build_slices(self, vgg: tv_models.VGG):
        features = vgg.features
        # Slice boundaries: relu1_2=3, relu2_2=8, relu3_3=15, relu4_3=22, relu5_3=29
        boundaries = [4, 9, 16, 23, 30]
        start = 0
        for end in boundaries:
            self.slices.append(nn.Sequential(*features[start:end]))
            start = end

    def forward(self, x: torch.Tensor):
        feats = []
        for s in self.slices:
            x = s(x)
            feats.append(x)
        return feats


class LPIPS(nn.Module):
    """
    Learned Perceptual Image Patch Similarity.

    Adapted from taming-transformers and the original LPIPS paper.
    Expects 3-channel input in approximately [-1, 1] range (will be normalized
    to ImageNet statistics internally).
    """

    def __init__(
        self,
        use_dropout: bool = True,
        pretrained: bool = True,
        lpips_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = VGG16Features.CHANNELS
        self.vgg = VGG16Features()

        self.lin = nn.ModuleList()
        for ch in self.chns:
            self.lin.append(NetLinLayer(ch, 1, use_dropout=use_dropout))

        if pretrained:
            self._load_pretrained_lin()

        # Freeze VGG, only train the lin layers
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.eval()

    def _load_pretrained_lin(self):
        """Load pretrained linear layer weights from the lpips package."""
        try:
            import lpips as _lpips

            ref = _lpips.LPIPS(net="vgg", verbose=False)
            if hasattr(ref, "lin"):
                ref_lins = ref.lin
            elif hasattr(ref, "lins"):
                ref_lins = ref.lins
            else:
                ref_lins = [getattr(ref, f"lin{i}", None) for i in range(len(self.chns))]

            for i, _ in enumerate(self.chns):
                ref_layer = ref_lins[i]
                if ref_layer is None:
                    raise AttributeError("LPIPS reference linear layer not found")

                if hasattr(ref_layer, "conv"):
                    w = ref_layer.conv.weight.data.clone()
                elif hasattr(ref_layer, "model") and len(ref_layer.model) > 0 and hasattr(ref_layer.model[-1], "weight"):
                    w = ref_layer.model[-1].weight.data.clone()
                else:
                    raise AttributeError("LPIPS reference linear layer has no conv weight")

                self.lin[i].conv.weight.data.copy_(w)
            del ref
        except ImportError:
            import warnings

            warnings.warn(
                "lpips package not found. Linear layer weights will be randomly initialized. "
                "Install with: pip install lpips",
                UserWarning,
            )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS distance between input and target.

        Args:
            input: (B, 3, H, W) tensor
            target: (B, 3, H, W) tensor

        Returns:
            (B,) tensor of perceptual distances
        """
        in_scaled = self.scaling_layer(input)
        tgt_scaled = self.scaling_layer(target)

        in_feats = self.vgg(in_scaled)
        tgt_feats = self.vgg(tgt_scaled)

        diffs = {}
        for k in range(len(self.chns)):
            diff = (in_feats[k] - tgt_feats[k]) ** 2
            # Normalize by channel dimension
            diff = diff / (2.0 * self.chns[k])  # approximate normalization
            diffs[k] = diff

        res = []
        for k in range(len(self.chns)):
            mapped = self.lin[k](diffs[k])
            res.append(mapped.squeeze(1).mean(dim=[1, 2]))  # (B,)

        val = torch.stack(res, dim=0).sum(dim=0)  # (B,)
        return val
