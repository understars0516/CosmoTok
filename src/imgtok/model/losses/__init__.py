from imgtok.model.losses.lpips import LPIPS
from imgtok.model.losses.discriminator import NLayerDiscriminator, ActNorm, weights_init
from imgtok.model.losses.vqperceptual import adopt_weight, hinge_d_loss, vanilla_d_loss
from imgtok.model.losses.astro_perceptual import AstroLPIPSWithDiscriminator, ChannelAdapter

__all__ = [
    "LPIPS",
    "NLayerDiscriminator",
    "ActNorm",
    "weights_init",
    "adopt_weight",
    "hinge_d_loss",
    "vanilla_d_loss",
    "AstroLPIPSWithDiscriminator",
    "ChannelAdapter",
]
