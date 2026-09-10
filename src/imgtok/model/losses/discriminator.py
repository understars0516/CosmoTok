import torch
import torch.nn as nn


def weights_init(m):
    """Initialize conv weights with N(0, 0.02) and batchnorm weights with N(1, 0.02)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class ActNorm(nn.Module):
    """
    Activation Normalization layer (like in Glow/StyleGAN).
    Replaces BatchNorm in the discriminator for stable training with small batches.
    """

    def __init__(self, num_features: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.register_parameter("bias", nn.Parameter(torch.zeros(1, num_features, 1, 1)))
        self.register_parameter("log_scale", nn.Parameter(torch.zeros(1, num_features, 1, 1)))
        self.initialized = False

    def initialize(self, x: torch.Tensor):
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True)
            self.bias.data.copy_(-mean)
            self.log_scale.data.copy_(torch.log(1.0 / (std + 1e-6)))
            self.initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized and self.training:
            self.initialize(x)
        return self.scale * torch.exp(self.log_scale) * x + self.bias


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator with configurable number of layers.

    Supports arbitrary input channel count (e.g., 9 for multi-band astronomical
    images, 1 for single-channel data).

    Args:
        input_nc: Number of input channels
        ndf: Base number of discriminator filters
        n_layers: Number of intermediate conv layers
        use_actnorm: If True, use ActNorm instead of BatchNorm
    """

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        use_actnorm: bool = False,
    ):
        super().__init__()
        norm_layer = ActNorm if use_actnorm else nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)

        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
