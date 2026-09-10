import torch
import torch.nn as nn
import torch.nn.functional as F

from imgtok.model.losses.discriminator import NLayerDiscriminator, weights_init
from imgtok.model.losses.vqperceptual import adopt_weight, hinge_d_loss, vanilla_d_loss


class ChannelAdapter(nn.Module):
    """
    Learnable 1x1 convolution that projects N-channel images to 3 channels
    for LPIPS perceptual loss computation.

    Astronomical images have 9 channels (HSC+DES multi-survey) or 1 channel
    (CosmoGrid), while LPIPS requires 3-channel RGB-like input. This adapter
    learns the optimal projection rather than using a fixed mapping.

    When in_channels >= 3, the adapter is initialized with an approximate
    identity mapping (first 3 channels pass through, others zeroed), providing
    a reasonable starting point before training.

    Args:
        in_channels: Number of input channels (e.g., 9 for multi-survey, 1 for CosmoGrid)
        out_channels: Number of output channels (default 3 for LPIPS)
    """

    def __init__(self, in_channels: int, out_channels: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

        # Initialize with near-identity mapping when possible
        if in_channels >= out_channels:
            with torch.no_grad():
                # First out_channels channels pass through approximately
                self.proj.weight.data[:out_channels, :out_channels] = (
                    torch.eye(out_channels).unsqueeze(-1).unsqueeze(-1)
                )
                self.proj.weight.data[:out_channels, out_channels:] = 0.0
                self.proj.bias.data.zero_()
        else:
            # For 1-channel input, simple expansion with small random weights
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AstroLPIPSWithDiscriminator(nn.Module):
    """
    Unified loss module for astronomical image autoencoder training with
    LPIPS perceptual loss and PatchGAN discriminator.

    Adapted from vavae's LPIPSWithDiscriminator, with these key differences:
    1. NO VF (vision foundation model) alignment — removed entirely
    2. ChannelAdapter for multi-channel -> 3-channel LPIPS mapping
    3. Masked reconstruction loss support (for multi-survey models)
    4. Inverse-variance weighting support (for ImageAutoEncoderV2)
    5. No KL divergence term (FSQ quantization, not VAE posterior)
    6. Variable discriminator input channels (9ch or 1ch)

    Loss formula for generator (optimizer_idx=0):
        loss = nll_loss + d_weight * disc_factor * g_loss

    Where:
        nll_loss = rec_loss_total / exp(logvar) + logvar
        rec_loss_total = pixel_loss + perceptual_weight * lpips_loss
        g_loss = -mean(D(reconstruction))  (non-saturating generator loss)
        d_weight = adaptive weight based on gradient norm ratio

    Loss formula for discriminator (optimizer_idx=1):
        d_loss = disc_factor * hinge_d_loss(D(real), D(fake))

    Args:
        disc_start: Step number to start discriminator training (warmup)
        logvar_init: Initial value for learnable log-variance
        pixelloss_weight: Weight for pixel reconstruction loss
        perceptual_weight: Weight for LPIPS perceptual loss
        disc_in_channels: Number of input channels for discriminator
        disc_num_layers: Number of intermediate layers in discriminator
        disc_ndf: Base number of discriminator filters
        disc_factor: Global discriminator scaling factor
        disc_weight: Base discriminator weight for adaptive calculation
        disc_loss: Discriminator loss type ("hinge" or "vanilla")
        use_actnorm: Whether to use ActNorm instead of BatchNorm in discriminator
        lpips_in_channels: Number of input channels for LPIPS ChannelAdapter
        lpips_out_channels: Number of output channels for LPIPS ChannelAdapter (default 3)
        use_mask: Whether to apply pixel mask in reconstruction loss
        use_ivar: Whether to use inverse-variance weighting in reconstruction loss
    """

    def __init__(
        self,
        disc_start: int = 5000,
        logvar_init: float = 0.0,
        pixelloss_weight: float = 1.0,
        perceptual_weight: float = 0.0,
        disc_in_channels: int = 9,
        disc_num_layers: int = 3,
        disc_ndf: int = 64,
        disc_factor: float = 1.0,
        disc_weight: float = 0.5,
        disc_loss: str = "hinge",
        use_actnorm: bool = False,
        lpips_in_channels: int = 9,
        lpips_out_channels: int = 3,
        use_mask: bool = True,
        use_ivar: bool = False,
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]

        # Pixel loss
        self.pixel_weight = pixelloss_weight

        # Perceptual loss (only initialized when perceptual_weight > 0)
        # LPIPS uses a VGG-16 trained on ImageNet (natural images), which may not
        # be suitable for astronomical images. Set perceptual_weight=0 (default)
        # to skip LPIPS entirely — no VGG download, no lpips package needed.
        self.perceptual_weight = perceptual_weight
        if perceptual_weight > 0:
            from imgtok.model.losses.lpips import LPIPS
            self.channel_adapter = ChannelAdapter(lpips_in_channels, lpips_out_channels)
            self.perceptual_loss = LPIPS().eval()
        else:
            self.channel_adapter = None
            self.perceptual_loss = None

        # Learnable log-variance for NLL weighting
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # Discriminator
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            ndf=disc_ndf,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss_fn = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

        # Masking/weighting flags
        self.use_mask = use_mask
        self.use_ivar = use_ivar

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Balance generator/discriminator gradients by computing the ratio
        of their gradient norms w.r.t. the last decoder layer.

        This prevents the discriminator from dominating training by scaling
        the adversarial loss to match the reconstruction loss gradient magnitude.

        Args:
            nll_loss: Reconstruction + perceptual loss
            g_loss: Generator adversarial loss
            last_layer: Weight tensor of the last decoder layer
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def _compute_pixel_loss(self, inputs, reconstructions, mask=None, ivar=None):
        """
        Compute pixel-level reconstruction loss.

        Three variants based on use_mask and use_ivar flags:
        - Masked, unweighted: mask * (input - recon)^2, sum / batch_size
        - Masked, ivar-weighted: ivar * mask * (input - recon)^2, sum / (num_pixels * batch_size)
        - Unmasked: (input - recon)^2, sum / batch_size
        """
        batch_size = inputs.shape[0]
        diff = inputs.contiguous() - reconstructions.contiguous()

        if self.use_mask and mask is not None:
            masked_diff = mask * diff
            if self.use_ivar and ivar is not None:
                _, _, h, w = inputs.shape
                num_pixels = h * w
                pixel_loss = (ivar * masked_diff * masked_diff).sum() / (num_pixels * batch_size)
            else:
                pixel_loss = (masked_diff * masked_diff).sum() / batch_size
        else:
            pixel_loss = (diff * diff).sum() / batch_size

        return self.pixel_weight * pixel_loss

    def forward(
        self,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        mask=None,
        ivar=None,
        split="train",
    ):
        """
        Compute loss for generator (optimizer_idx=0) or discriminator (optimizer_idx=1).

        Args:
            inputs: Original images (B, C, H, W)
            reconstructions: Reconstructed images (B, C, H, W)
            optimizer_idx: 0 for generator (AE), 1 for discriminator
            global_step: Current training step number
            last_layer: Last decoder layer weight for adaptive weight calculation
            mask: Binary mask for valid pixels (B, C, H, W), optional
            ivar: Inverse variance weights (B, C, H, W), optional
            split: "train" or "val" for logging prefix
        """
        # ---- Pixel + Perceptual loss (shared for both optimizer_idx) ----
        pixel_loss = self._compute_pixel_loss(inputs, reconstructions, mask, ivar)

        # LPIPS perceptual loss (skipped when perceptual_weight == 0)
        if self.perceptual_weight > 0 and self.perceptual_loss is not None:
            inputs_3ch = self.channel_adapter(inputs.contiguous())
            recon_3ch = self.channel_adapter(reconstructions.contiguous())
            p_loss = self.perceptual_loss(inputs_3ch, recon_3ch).mean()
        else:
            p_loss = torch.tensor(0.0, device=inputs.device)

        rec_loss_total = pixel_loss + self.perceptual_weight * p_loss

        # NLL loss with learnable log-variance
        nll_loss = rec_loss_total / torch.exp(self.logvar) + self.logvar

        # ---- Generator update (optimizer_idx == 0) ----
        if optimizer_idx == 0:
            # Generator adversarial loss
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)

            # Adaptive discriminator weight
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            if disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            loss = nll_loss + d_weight * disc_factor * g_loss

            log = {
                f"{split}/total_loss": loss.detach().mean(),
                f"{split}/logvar": self.logvar.detach(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss_total.detach().mean(),
                f"{split}/pixel_loss": pixel_loss.detach().mean(),
                f"{split}/p_loss": p_loss.detach().mean(),
                f"{split}/d_weight": d_weight.detach() if isinstance(d_weight, torch.Tensor) else torch.tensor(d_weight),
                f"{split}/disc_factor": torch.tensor(disc_factor),
                f"{split}/g_loss": g_loss.detach().mean(),
            }
            return loss, log

        # ---- Discriminator update (optimizer_idx == 1) ----
        if optimizer_idx == 1:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss_fn(logits_real, logits_fake)

            log = {
                f"{split}/disc_loss": d_loss.detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean(),
            }
            return d_loss, log