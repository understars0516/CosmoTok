import math
import typing
from abc import ABCMeta, abstractmethod
from functools import partial

import lightning as ltn
import lightning.pytorch as pl
import numpy as np
import torch
import torchmetrics
from imgtok.model.codec import AutoencoderCosmoGridImageCodec, ImageCodec, ImageCodecV2
from imgtok.model.losses.astro_perceptual import AstroLPIPSWithDiscriminator
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

if typing.TYPE_CHECKING:
    from typing import Literal, Sequence

    from imgtok.common import ImageDatum
    from lightning.pytorch.utilities.types import OptimizerLRScheduler
    from torch import Tensor


class RequiresQuantizerModelMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        # 创建实例
        instance = super().__call__(*args, **kwargs)

        # 检查必需字段
        if not hasattr(instance, "model"):
            raise AttributeError(
                f"{cls.__name__} instance must have a field: 'model'"
            )

        if not hasattr(instance.model, "quantizer"):
            raise AttributeError(
                f"{cls.__name__} self.model must have a field: 'quantizer'"
            )

        return instance


class FSQAutoEncoderLightningModule(ltn.LightningModule, metaclass=RequiresQuantizerModelMeta):
    def __init__(self, *args,
                 fsq_annealing_method: "Literal['cosine', 'linear', 'none']" = None,
                 fsq_annealing_factor: float = 0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if fsq_annealing_method == "none" or fsq_annealing_factor <= 0.0:
            fsq_annealing_method = None

        self.annealing_factor = fsq_annealing_factor
        match fsq_annealing_method:
            case "linear":
                self.fsq_annealing = self.linear_schedule
            case "cosine":
                self.fsq_annealing = self.cosine_schedule
            case None:
                self.fsq_annealing = None

    def training_step(self, batch, batch_idx):
        # 计算损失
        loss = self.loss_fn(batch)
        
        # 检查损失值是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            self.trainer.should_stop = True
            raise ValueError(f"Loss is invalid: {loss.item()}. Stopping training.")
        
        self.log("train_loss", loss.item(), prog_bar=True)
        return loss

    def on_train_start(self):
        # 打印日志目录 (例如: lightning_logs/version_0)
        print(f"\nlogs are saved to: {self.logger.save_dir}")

        # 打印 Checkpoint 目录
        # noinspection PyUnresolvedReferences
        for callback in self.trainer.callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                print(f"checkpoint directory: {callback.dirpath}")

    def on_fit_start(self) -> None:
        if self.fsq_annealing is None:
            self.model.quantizer.set_no_quant_prob(0.0)

    def on_train_batch_start(self, batch, batch_idx):
        if self.fsq_annealing is None:
            p = self.model.quantizer.get_no_quant_prob()
            self.log("no_quant_prob", p)
            return

        p = self.fsq_annealing()
        p = self.model.quantizer.set_no_quant_prob(p)

        # 记录到日志以便监控
        self.log("no_quant_prob", p)

    @abstractmethod
    def loss_fn(self, x):
        raise NotImplementedError("loss function not implemented for abstract class")

    def cosine_schedule(self):
        current_step = self.global_step
        total_steps = self.trainer.estimated_stepping_batches
        progress = current_step / (self.annealing_factor * total_steps)
        progress = np.clip(progress, 0.0, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def linear_schedule(self):
        current_step = self.global_step
        total_steps = self.trainer.estimated_stepping_batches
        progress = current_step / (self.annealing_factor * total_steps)
        return np.clip(progress, 0.0, 1.0)


class ImageAutoEncoder(FSQAutoEncoderLightningModule):
    def __init__(
            self,
            quantizer_levels: "Sequence[int]" = (7, 5, 5, 5, 5),
            hidden_dims: int = 512,
            embedding_dim: int = 5,
            multisurvey_projection_dims: int = 54,
            n_compressions: int = 2,
            num_consecutive: int = 4,
            range_compression_factor: float = 0.01,
            mult_factor: float = 10.0,
            warmup_steps: int = 1000,
            fsq_annealing_method: "Literal['cosine', 'linear', 'none']" = None,
            fsq_annealing_factor: float = 0.0,
    ) -> None:
        """
        Initializes the Image AutoEncoder model.

        Args:
            quantizer_levels (Sequence[int]): Quantizer levels, defaults to (7, 5, 5, 5, 5)
            hidden_dims (int): Hidden dimensions, defaults to 512
            embedding_dim (int): Embedding dimension, defaults to 5
            multisurvey_projection_dims (int): Multi-survey projection dimensions, defaults to 54
            n_compressions (int): Number of compressions, defaults to 2
            num_consecutive (int): Number of consecutive operations, defaults to 4
            range_compression_factor (float): Range compression factor, defaults to 0.01
            mult_factor (float): Multiplication factor, defaults to 10.0
            warmup_steps (int): Number of warmup steps, defaults to 5000
        """
        super().__init__(
            fsq_annealing_factor=fsq_annealing_factor,
            fsq_annealing_method=fsq_annealing_method,
        )
        self.model = ImageCodec(
            quantizer_levels=quantizer_levels,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            multisurvey_projection_dims=multisurvey_projection_dims,
            n_compressions=n_compressions,
            num_consecutive=num_consecutive,
            range_compression_factor=range_compression_factor,
            mult_factor=mult_factor,
        )
        self.warmup_steps = warmup_steps
        
        # Initialize MSE metric for validation
        self.val_mse = torchmetrics.MeanSquaredError()
        
        self.save_hyperparameters()

    # noinspection DuplicatedCode
    def validation_step(self, batch: "ImageDatum", batch_idx):
        # Calculate MSE between predicted and actual flux
        flux, mask, channel_mask = batch.flux, batch.mask, batch.channel_mask  # (B, C, H, W)
        batch_size = flux.shape[0]

        recon_flux = self.model(flux, channel_mask)  # (B, C, H, W)
        recon_flux = recon_flux[mask]  # (N, ), where N = sum(mask)
        valid_flux = flux[mask]

        # Update MSE metric
        # noinspection DuplicatedCode
        self.val_mse.update(recon_flux.detach().flatten(), valid_flux.detach().flatten())

        # Log the MSE metric
        self.log("val_mse", self.val_mse, batch_size=batch_size, prog_bar=True)

        # this is the test loop
        val_loss = self.loss_fn(batch).item()
        self.log("val_loss", val_loss, batch_size=batch_size, prog_bar=True)

    def loss_fn(self, x: "ImageDatum"):
        batch_size = x.flux.shape[0]

        weighted_diff = x.mask * (x.flux - self.model(x.flux, x.channel_mask))
        return (weighted_diff * weighted_diff).sum() / batch_size

    # noinspection DuplicatedCode
    def configure_optimizers(self):
        r"""Choose what optimizers and learning-rate schedulers to use in your optimization. Normally you'd need one.
            But in the case of GANs or similar you might have multiple. Optimization with multiple optimizers only works
            in the manual optimization mode.

            Return:
                Any of these 6 options.

                - **Single optimizer**.
                - **List or Tuple** of optimizers.
                - **Two lists** - The first list has multiple optimizers, and the second has multiple LR schedulers
                  (or multiple ``lr_scheduler_config``).
                - **Dictionary**, with an ``"optimizer"`` key, and (optionally) a ``"lr_scheduler"``
                  key whose value is a single LR scheduler or ``lr_scheduler_config``.
                - **None** - Fit will run without any optimizer.

            The ``lr_scheduler_config`` is a dictionary which contains the scheduler and its associated configuration.
            The default configuration is shown below.

            .. code-block:: python

                lr_scheduler_config = {
                    # REQUIRED: The scheduler instance
                    "scheduler": lr_scheduler,
                    # The unit of the scheduler's step size, could also be 'step'.
                    # 'epoch' updates the scheduler on epoch end whereas 'step'
                    # updates it after a optimizer update.
                    "interval": "epoch",
                    # How many epochs/steps should pass between calls to
                    # `scheduler.step()`. 1 corresponds to updating the learning
                    # rate after every epoch/step.
                    "frequency": 1,
                    # Metric to monitor for schedulers like `ReduceLROnPlateau`
                    "monitor": "val_loss",
                    # If set to `True`, will enforce that the value specified 'monitor'
                    # is available when the scheduler is updated, thus stopping
                    # training if not found. If set to `False`, it will only produce a warning
                    "strict": True,
                    # If using the `LearningRateMonitor` callback to monitor the
                    # learning rate progress, this keyword can be used to specify
                    # a custom logged name
                    "name": None,
                }

            When there are schedulers in which the ``.step()`` method is conditioned on a value, such as the
            :class:`torch.optim.lr_scheduler.ReduceLROnPlateau` scheduler, Lightning requires that the
            ``lr_scheduler_config`` contains the keyword ``"monitor"`` set to the metric name that the scheduler
            should be conditioned on.

            .. testcode::

                # The ReduceLROnPlateau scheduler requires a monitor
                def configure_optimizers(self):
                    optimizer = Adam(...)
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": ReduceLROnPlateau(optimizer, ...),
                            "monitor": "metric_to_track",
                            "frequency": "indicates how often the metric is updated",
                            # If "monitor" references validation metrics, then "frequency" should be set to a
                            # multiple of "trainer.check_val_every_n_epoch".
                        },
                    }


                # In the case of two optimizers, only one using the ReduceLROnPlateau scheduler
                def configure_optimizers(self):
                    optimizer1 = Adam(...)
                    optimizer2 = SGD(...)
                    scheduler1 = ReduceLROnPlateau(optimizer1, ...)
                    scheduler2 = LambdaLR(optimizer2, ...)
                    return (
                        {
                            "optimizer": optimizer1,
                            "lr_scheduler": {
                                "scheduler": scheduler1,
                                "monitor": "metric_to_track",
                            },
                        },
                        {"optimizer": optimizer2, "lr_scheduler": scheduler2},
                    )

            Metrics can be made available to monitor by simply logging it using
            ``self.log('metric_to_track', metric_val)`` in your :class:`~lightning.pytorch.core.LightningModule`.

            Note:
                Some things to know:

                - Lightning calls ``.backward()`` and ``.step()`` automatically in case of automatic optimization.
                - If a learning rate scheduler is specified in ``configure_optimizers()`` with key
                  ``"interval"`` (default "epoch") in the scheduler configuration, Lightning will call
                  the scheduler's ``.step()`` method automatically in case of automatic optimization.
                - If you use 16-bit precision (``precision=16``), Lightning will automatically handle the optimizer.
                - If you use :class:`torch.optim.LBFGS`, Lightning handles the closure function automatically for you.
                - If you use multiple optimizers, you will have to switch to 'manual optimization' mode and step them
                  yourself.
                - If you need to control how often the optimizer steps, override the :meth:`optimizer_step` hook.

            """
        
        optimizer = optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-5)

        total_steps = self.trainer.estimated_stepping_batches
        # print(f"==== Debug: got total steps: {total_steps}, {total_steps // self.trainer.max_epochs} per epoch")
        schedule_fn = partial(warmup_cosine_schedule, warmup_steps=self.warmup_steps, total_steps=total_steps)
        scheduler = LambdaLR(optimizer, lr_lambda=schedule_fn)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update learning rate every step, not epoch
            }
        }


class ImageAutoEncoderV2(FSQAutoEncoderLightningModule):
    def __init__(
            self,
            quantizer_levels: "Sequence[int]" = (7, 5, 5, 5, 5),
            hidden_dims: int = 512,
            embedding_dim: int = 5,
            multisurvey_projection_dims: int = 54,
            n_compressions: int = 2,
            num_consecutive: int = 4,
            warmup_steps: int = 1000,
            fsq_annealing_method: "Literal['cosine', 'linear', 'none']" = None,
            fsq_annealing_factor: float = 0.0,
    ) -> None:
        """
        Initializes the Image AutoEncoder model.

        Args:
            quantizer_levels (Sequence[int]): Quantizer levels, defaults to (7, 5, 5, 5, 5)
            hidden_dims (int): Hidden dimensions, defaults to 512
            embedding_dim (int): Embedding dimension, defaults to 5
            multisurvey_projection_dims (int): Multi-survey projection dimensions, defaults to 54
            n_compressions (int): Number of compressions, defaults to 2
            num_consecutive (int): Number of consecutive operations, defaults to 4
            warmup_steps (int): Number of warmup steps, defaults to 5000
        """
        super().__init__(
            fsq_annealing_method=fsq_annealing_method,
            fsq_annealing_factor=fsq_annealing_factor,
        )
        self.model = ImageCodecV2(
            quantizer_levels=quantizer_levels,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            multisurvey_projection_dims=multisurvey_projection_dims,
            n_compressions=n_compressions,
            num_consecutive=num_consecutive,
        )
        self.warmup_steps = warmup_steps

        # Initialize MSE metric for validation
        self.val_mse = torchmetrics.MeanSquaredError()

        self.save_hyperparameters()

    # noinspection DuplicatedCode
    def validation_step(self, batch: "ImageDatum", batch_idx):
        # Calculate MSE between predicted and actual flux
        flux, mask, channel_mask = batch.flux, batch.mask, batch.channel_mask  # (B, C, H, W)
        batch_size = flux.shape[0]

        recon_flux = self.model(flux, channel_mask)  # (B, C, H, W)
        recon_flux = recon_flux[mask]  # (N, ), where N = sum(mask)
        valid_flux = flux[mask]

        # Update MSE metric
        # noinspection DuplicatedCode
        self.val_mse.update(recon_flux.detach().flatten(), valid_flux.detach().flatten())

        # Log the MSE metric
        self.log("val_mse", self.val_mse, batch_size=batch_size, prog_bar=True)

        # this is the test loop
        val_loss = self.loss_fn(batch).item()
        self.log("val_loss", val_loss, batch_size=batch_size, prog_bar=True)

    def loss_fn(self, x: "ImageDatum"):
        batch_size, _, *spatial_size = x.flux.shape
        num_pixels = spatial_size[0] * spatial_size[1]  # 被掩码比例非常小, 因此可以不用 mask.sum() 来做平均.

        weight = x.ivar if x.ivar is not None else 1.0
        masked_diff = x.mask * (x.flux - self.model(x.flux, x.channel_mask))
        return (weight * masked_diff * masked_diff).sum() / (num_pixels * batch_size)

    # noinspection DuplicatedCode
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-5)

        total_steps = self.trainer.estimated_stepping_batches
        # print(f"==== Debug: got total steps: {total_steps}, {total_steps // self.trainer.max_epochs} per epoch")
        schedule_fn = partial(warmup_cosine_schedule, warmup_steps=self.warmup_steps, total_steps=total_steps)
        scheduler = LambdaLR(optimizer, lr_lambda=schedule_fn)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update learning rate every step, not epoch
            }
        }


class CosmoGridImageAutoEncoder(FSQAutoEncoderLightningModule):
    def __init__(
            self,
            hidden_dims: int = 256,
            embedding_dim: int = 5,
            downsampled_channel_dim: int = 16,
            quantizer_levels: "Sequence[int]" = (7, 7, 5, 5, 5),
            n_compressions: int = 2,
            num_consecutive: int = 4,
            warmup_steps: int = 1000,
            fsq_annealing_method: "Literal['cosine', 'linear', 'none']" = None,
            fsq_annealing_factor: float = 0.0,
    ):
        super().__init__(
            fsq_annealing_method=fsq_annealing_method,
            fsq_annealing_factor=fsq_annealing_factor,
        )
        self.save_hyperparameters()

        self.model = AutoencoderCosmoGridImageCodec(
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            downsampled_channel_dim=downsampled_channel_dim,
            quantizer_levels=quantizer_levels,
            n_compressions=n_compressions,
            num_consecutive=num_consecutive,
        )
        self.warmup_steps = warmup_steps

        # Initialize MSE metric for validation
        self.val_mse = torchmetrics.MeanSquaredError()

    def loss_fn(self, x):
        batch_size = x.shape[0]

        diff = x - self.model(x)
        return (diff * diff).sum() / batch_size

    def validation_step(self, x: "Tensor", batch_idx):
        batch_size = x.shape[0]

        recon = self.model(x)  # (B, C, H, W)

        # Update MSE metric
        # noinspection DuplicatedCode
        self.val_mse.update(recon.detach().flatten(), x.detach().flatten())

        # Log the MSE metric
        self.log("val_mse", self.val_mse, batch_size=batch_size, prog_bar=True)

        # this is the test loop
        val_loss = self.loss_fn(x).item()
        self.log("val_loss", val_loss, batch_size=batch_size, prog_bar=True)

    # noinspection DuplicatedCode
    def configure_optimizers(self) -> "OptimizerLRScheduler":
        # noinspection DuplicatedCode
        optimizer = optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-5)

        total_steps = self.trainer.estimated_stepping_batches
        schedule_fn = partial(warmup_cosine_schedule, warmup_steps=self.warmup_steps, total_steps=total_steps)
        scheduler = LambdaLR(optimizer, lr_lambda=schedule_fn)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update learning rate every step, not epoch
            }
        }


def warmup_cosine_schedule(current_step: int, warmup_steps: int, total_steps: int):
    if current_step < warmup_steps:
        # Linear warmup
        return float(current_step) / float(max(1, warmup_steps))
    else:
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))


# ============================================================================
# Adversarial training variants with LPIPS perceptual loss + PatchGAN discriminator
# ============================================================================


class ImageAutoEncoderAdv(FSQAutoEncoderLightningModule):
    """
    Adversarial variant of ImageAutoEncoder with LPIPS perceptual loss and
    PatchGAN discriminator.

    Uses manual optimization with two optimizers:
    - AE optimizer: AdamW for encoder + decoder + channel_adapter + logvar
    - Discriminator optimizer: Adam for PatchGAN discriminator

    The discriminator operates on the full 9-channel multi-survey output.
    LPIPS uses a learnable ChannelAdapter (9ch -> 3ch) for perceptual loss.
    """

    def __init__(
            self,
            quantizer_levels: "Sequence[int]" = (7, 5, 5, 5, 5),
            hidden_dims: int = 512,
            embedding_dim: int = 5,
            multisurvey_projection_dims: int = 54,
            n_compressions: int = 2,
            num_consecutive: int = 4,
            range_compression_factor: float = 0.01,
            mult_factor: float = 10.0,
            warmup_steps: int = 1000,
            fsq_annealing_method: "Literal['cosine', 'linear', 'none']" = None,
            fsq_annealing_factor: float = 0.0,
            # Adversarial / perceptual parameters
            disc_start: int = 5000,
            disc_in_channels: int = 9,
            disc_num_layers: int = 3,
            disc_ndf: int = 64,
            disc_factor: float = 1.0,
            disc_weight: float = 0.5,
            disc_loss: str = "hinge",
            perceptual_weight: float = 1.0,
            pixelloss_weight: float = 1.0,
            logvar_init: float = 0.0,
            lpips_in_channels: int = 9,
            use_actnorm: bool = False,
            disc_lr: float = 1e-4,
            ae_lr: float = 5e-4,
    ) -> None:
        super().__init__(
            fsq_annealing_factor=fsq_annealing_factor,
            fsq_annealing_method=fsq_annealing_method,
        )
        self.automatic_optimization = False

        self.model = ImageCodec(
            quantizer_levels=quantizer_levels,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            multisurvey_projection_dims=multisurvey_projection_dims,
            n_compressions=n_compressions,
            num_consecutive=num_consecutive,
            range_compression_factor=range_compression_factor,
            mult_factor=mult_factor,
        )
        self.warmup_steps = warmup_steps

        self.loss_module = AstroLPIPSWithDiscriminator(
            disc_start=disc_start,
            disc_in_channels=disc_in_channels,
            disc_num_layers=disc_num_layers,
            disc_ndf=disc_ndf,
            disc_factor=disc_factor,
            disc_weight=disc_weight,
            disc_loss=disc_loss,
            perceptual_weight=perceptual_weight,
            pixelloss_weight=pixelloss_weight,
            logvar_init=logvar_init,
            lpips_in_channels=lpips_in_channels,
            use_actnorm=use_actnorm,
            use_mask=True,
            use_ivar=False,
        )

        self.disc_lr = disc_lr
        self.ae_lr = ae_lr

        self.val_mse = torchmetrics.MeanSquaredError()
        self.save_hyperparameters()

    def get_last_layer(self):
        """Return the last decoder conv weight for adaptive weight calculation."""
        return self.model.ae.conv_out.conv.weight

    def loss_fn(self, x: "ImageDatum"):
        """Simplified loss for validation/monitoring (no discriminator)."""
        batch_size = x.flux.shape[0]
        flux, mask, channel_mask = x.flux, x.mask, x.channel_mask
        reconstructions = self.model(flux, channel_mask)

        # Pixel loss
        masked_diff = mask * (flux - reconstructions)
        pixel_loss = (masked_diff * masked_diff).sum() / batch_size

        # Perceptual loss (only if enabled)
        if self.loss_module.perceptual_weight > 0 and self.loss_module.perceptual_loss is not None:
            inputs_3ch = self.loss_module.channel_adapter(flux)
            recon_3ch = self.loss_module.channel_adapter(reconstructions)
            p_loss = self.loss_module.perceptual_loss(inputs_3ch, recon_3ch).mean()
            return pixel_loss + self.loss_module.perceptual_weight * p_loss
        return pixel_loss

    def training_step(self, batch, batch_idx):
        ae_opt, disc_opt = self.optimizers()

        flux, mask, channel_mask = batch.flux, batch.mask, batch.channel_mask
        reconstructions = self.model(flux, channel_mask)

        # --- Generator (AE) step ---
        aeloss, log_dict_ae = self.loss_module(
            inputs=flux, reconstructions=reconstructions,
            optimizer_idx=0, global_step=self.global_step,
            last_layer=self.get_last_layer(), mask=mask, split="train",
        )

        if torch.isnan(aeloss) or torch.isinf(aeloss):
            self.trainer.should_stop = True
            raise ValueError(f"AE loss is invalid: {aeloss.item()}. Stopping training.")

        ae_opt.zero_grad()
        self.manual_backward(aeloss)
        ae_opt.step()

        # --- Discriminator step ---
        discloss, log_dict_disc = self.loss_module(
            inputs=flux, reconstructions=reconstructions,
            optimizer_idx=1, global_step=self.global_step,
            last_layer=self.get_last_layer(), mask=mask, split="train",
        )

        disc_opt.zero_grad()
        self.manual_backward(discloss)
        disc_opt.step()

        # Logging
        self.log("train_ae_loss", aeloss.item(), prog_bar=True)
        self.log("train_disc_loss", discloss.item(), prog_bar=True)
        self.log_dict(log_dict_ae, prog_bar=False, on_step=True, on_epoch=False)
        self.log_dict(log_dict_disc, prog_bar=False, on_step=True, on_epoch=False)

    # noinspection DuplicatedCode
    def validation_step(self, batch: "ImageDatum", batch_idx):
        flux, mask, channel_mask = batch.flux, batch.mask, batch.channel_mask
        batch_size = flux.shape[0]

        reconstructions = self.model(flux, channel_mask)

        # Update MSE metric
        recon_masked = reconstructions[mask]
        valid_flux = flux[mask]
        self.val_mse.update(recon_masked.detach().flatten(), valid_flux.detach().flatten())
        self.log("val_mse", self.val_mse, batch_size=batch_size, prog_bar=True)

        # Compute losses for logging
        aeloss, log_dict_ae = self.loss_module(
            inputs=flux, reconstructions=reconstructions,
            optimizer_idx=0, global_step=self.global_step,
            last_layer=self.get_last_layer(), mask=mask, split="val",
        )
        discloss, log_dict_disc = self.loss_module(
            inputs=flux, reconstructions=reconstructions,
            optimizer_idx=1, global_step=self.global_step,
            last_layer=self.get_last_layer(), mask=mask, split="val",
        )

        self.log("val_loss", aeloss.item(), batch_size=batch_size, prog_bar=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

    # noinspection DuplicatedCode
    def configure_optimizers(self):
        # AE optimizer: model + channel_adapter (if LPIPS enabled) + logvar
        ae_params = list(self.model.parameters())
        if self.loss_module.channel_adapter is not None:
            ae_params += list(self.loss_module.channel_adapter.parameters())
        ae_params.append(self.loss_module.logvar)
        ae_opt = optim.AdamW(ae_params, lr=self.ae_lr, weight_decay=1e-5)

        # Discriminator optimizer
        disc_opt = optim.Adam(
            self.loss_module.discriminator.parameters(),
            lr=self.disc_lr, betas=(0.5, 0.9),
        )

        # AE scheduler: warmup + cosine
        total_steps = self.trainer.estimated_stepping_batches
        schedule_fn = partial(warmup_cosine_schedule, warmup_steps=self.warmup_steps, total_steps=total_steps)
        ae_scheduler = LambdaLR(ae_opt, lr_lambda=schedule_fn)

        return [
            {
                "optimizer": ae_opt,
                "lr_scheduler": {"scheduler": ae_scheduler, "interval": "step"},
            },
            {"optimizer": disc_opt},
        ]


class ImageAutoEncoderV2Adv(FSQAutoEncoderLightningModule):
    """
    Adversarial variant of ImageAutoEncoderV2 with LPIPS perceptual loss,
    PatchGAN discriminator, and inverse-variance weighted reconstruction loss.
    """

    def __init__(
            self,
            quantizer_levels: "Sequence[int]" = (7, 5, 5, 5, 5),
            hidden_dims: int = 512,
            embedding_dim: int = 5,
            multisurvey_projection_dims: int = 54,
            n_compressions: int = 2,
            num_consecutive: int = 4,
            warmup_steps: int = 1000,
            fsq_annealing_method: "Literal['cosine', 'linear', 'none']" = None,
            fsq_annealing_factor: float = 0.0,
            # Adversarial / perceptual parameters
            disc_start: int = 5000,
            disc_in_channels: int = 9,
            disc_num_layers: int = 3,
            disc_ndf: int = 64,
            disc_factor: float = 1.0,
            disc_weight: float = 0.5,
            disc_loss: str = "hinge",
            perceptual_weight: float = 1.0,
            pixelloss_weight: float = 1.0,
            logvar_init: float = 0.0,
            lpips_in_channels: int = 9,
            use_actnorm: bool = False,
            disc_lr: float = 1e-4,
            ae_lr: float = 5e-4,
    ) -> None:
        super().__init__(
            fsq_annealing_method=fsq_annealing_method,
            fsq_annealing_factor=fsq_annealing_factor,
        )
        self.automatic_optimization = False

        self.model = ImageCodecV2(
            quantizer_levels=quantizer_levels,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            multisurvey_projection_dims=multisurvey_projection_dims,
            n_compressions=n_compressions,
            num_consecutive=num_consecutive,
        )
        self.warmup_steps = warmup_steps

        self.loss_module = AstroLPIPSWithDiscriminator(
            disc_start=disc_start,
            disc_in_channels=disc_in_channels,
            disc_num_layers=disc_num_layers,
            disc_ndf=disc_ndf,
            disc_factor=disc_factor,
            disc_weight=disc_weight,
            disc_loss=disc_loss,
            perceptual_weight=perceptual_weight,
            pixelloss_weight=pixelloss_weight,
            logvar_init=logvar_init,
            lpips_in_channels=lpips_in_channels,
            use_actnorm=use_actnorm,
            use_mask=True,
            use_ivar=True,
        )

        self.disc_lr = disc_lr
        self.ae_lr = ae_lr

        self.val_mse = torchmetrics.MeanSquaredError()
        self.save_hyperparameters()

    def get_last_layer(self):
        """Return the last decoder conv weight for adaptive weight calculation."""
        return self.model.ae.conv_out.conv.weight

    def loss_fn(self, x: "ImageDatum"):
        """Simplified loss for validation/monitoring (no discriminator)."""
        batch_size, _, *spatial_size = x.flux.shape
        num_pixels = spatial_size[0] * spatial_size[1]
        flux, mask, channel_mask = x.flux, x.mask, x.channel_mask
        ivar = x.ivar if x.ivar is not None else 1.0
        reconstructions = self.model(flux, channel_mask)

        # Pixel loss (ivar-weighted)
        masked_diff = mask * (flux - reconstructions)
        pixel_loss = (ivar * masked_diff * masked_diff).sum() / (num_pixels * batch_size)

        # Perceptual loss (only if enabled)
        if self.loss_module.perceptual_weight > 0 and self.loss_module.perceptual_loss is not None:
            inputs_3ch = self.loss_module.channel_adapter(flux)
            recon_3ch = self.loss_module.channel_adapter(reconstructions)
            p_loss = self.loss_module.perceptual_loss(inputs_3ch, recon_3ch).mean()
            return pixel_loss + self.loss_module.perceptual_weight * p_loss
        return pixel_loss

    def training_step(self, batch, batch_idx):
        ae_opt, disc_opt = self.optimizers()

        flux, mask, channel_mask = batch.flux, batch.mask, batch.channel_mask
        ivar = batch.ivar
        reconstructions = self.model(flux, channel_mask)

        # --- Generator (AE) step ---
        aeloss, log_dict_ae = self.loss_module(
            inputs=flux, reconstructions=reconstructions,
            optimizer_idx=0, global_step=self.global_step,
            last_layer=self.get_last_layer(), mask=mask, ivar=ivar, split="train",
        )

        if torch.isnan(aeloss) or torch.isinf(aeloss):
            self.trainer.should_stop = True
            raise ValueError(f"AE loss is invalid: {aeloss.item()}. Stopping training.")

        ae_opt.zero_grad()
        self.manual_backward(aeloss)
        ae_opt.step()

        # --- Discriminator step ---
        discloss, log_dict_disc = self.loss_module(
            inputs=flux, reconstructions=reconstructions,
            optimizer_idx=1, global_step=self.global_step,
            last_layer=self.get_last_layer(), mask=mask, ivar=ivar, split="train",
        )

        disc_opt.zero_grad()
        self.manual_backward(discloss)
        disc_opt.step()

        # Logging
        self.log("train_ae_loss", aeloss.item(), prog_bar=True)
        self.log("train_disc_loss", discloss.item(), prog_bar=True)
        self.log_dict(log_dict_ae, prog_bar=False, on_step=True, on_epoch=False)
        self.log_dict(log_dict_disc, prog_bar=False, on_step=True, on_epoch=False)

    # noinspection DuplicatedCode
    def validation_step(self, batch: "ImageDatum", batch_idx):
        flux, mask, channel_mask = batch.flux, batch.mask, batch.channel_mask
        ivar = batch.ivar
        batch_size = flux.shape[0]

        reconstructions = self.model(flux, channel_mask)

        # Update MSE metric
        recon_masked = reconstructions[mask]
        valid_flux = flux[mask]
        self.val_mse.update(recon_masked.detach().flatten(), valid_flux.detach().flatten())
        self.log("val_mse", self.val_mse, batch_size=batch_size, prog_bar=True)

        # Compute losses for logging
        aeloss, log_dict_ae = self.loss_module(
            inputs=flux, reconstructions=reconstructions,
            optimizer_idx=0, global_step=self.global_step,
            last_layer=self.get_last_layer(), mask=mask, ivar=ivar, split="val",
        )
        discloss, log_dict_disc = self.loss_module(
            inputs=flux, reconstructions=reconstructions,
            optimizer_idx=1, global_step=self.global_step,
            last_layer=self.get_last_layer(), mask=mask, ivar=ivar, split="val",
        )

        self.log("val_loss", aeloss.item(), batch_size=batch_size, prog_bar=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

    # noinspection DuplicatedCode
    def configure_optimizers(self):
        ae_params = list(self.model.parameters())
        if self.loss_module.channel_adapter is not None:
            ae_params += list(self.loss_module.channel_adapter.parameters())
        ae_params.append(self.loss_module.logvar)
        ae_opt = optim.AdamW(ae_params, lr=self.ae_lr, weight_decay=1e-5)

        disc_opt = optim.Adam(
            self.loss_module.discriminator.parameters(),
            lr=self.disc_lr, betas=(0.5, 0.9),
        )

        total_steps = self.trainer.estimated_stepping_batches
        schedule_fn = partial(warmup_cosine_schedule, warmup_steps=self.warmup_steps, total_steps=total_steps)
        ae_scheduler = LambdaLR(ae_opt, lr_lambda=schedule_fn)

        return [
            {
                "optimizer": ae_opt,
                "lr_scheduler": {"scheduler": ae_scheduler, "interval": "step"},
            },
            {"optimizer": disc_opt},
        ]


class CosmoGridImageAutoEncoderAdv(FSQAutoEncoderLightningModule):
    """
    Adversarial variant of CosmoGridImageAutoEncoder with LPIPS perceptual loss
    and PatchGAN discriminator for single-channel cosmological simulation images.

    The discriminator operates on 1-channel images.
    LPIPS uses a learnable ChannelAdapter (1ch -> 3ch).
    """

    def __init__(
            self,
            hidden_dims: int = 256,
            embedding_dim: int = 5,
            downsampled_channel_dim: int = 16,
            quantizer_levels: "Sequence[int]" = (7, 7, 5, 5, 5),
            n_compressions: int = 2,
            num_consecutive: int = 4,
            warmup_steps: int = 1000,
            fsq_annealing_method: "Literal['cosine', 'linear', 'none']" = None,
            fsq_annealing_factor: float = 0.0,
            # Adversarial / perceptual parameters
            disc_start: int = 5000,
            disc_in_channels: int = 1,
            disc_num_layers: int = 3,
            disc_ndf: int = 64,
            disc_factor: float = 1.0,
            disc_weight: float = 0.5,
            disc_loss: str = "hinge",
            perceptual_weight: float = 1.0,
            pixelloss_weight: float = 1.0,
            logvar_init: float = 0.0,
            lpips_in_channels: int = 1,
            use_actnorm: bool = False,
            disc_lr: float = 1e-4,
            ae_lr: float = 5e-4,
            ae_betas: "Sequence[float]" = (0.9, 0.999),
            ae_weight_decay: float = 1e-5,
            ae_scheduler_type: str = "warmup_cosine",
            ae_scheduler_t_max: int = 1000,
            ae_scheduler_eta_min: float = 1e-6,
    ):
        super().__init__(
            fsq_annealing_method=fsq_annealing_method,
            fsq_annealing_factor=fsq_annealing_factor,
        )
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.model = AutoencoderCosmoGridImageCodec(
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            downsampled_channel_dim=downsampled_channel_dim,
            quantizer_levels=quantizer_levels,
            n_compressions=n_compressions,
            num_consecutive=num_consecutive,
        )
        self.warmup_steps = warmup_steps

        self.loss_module = AstroLPIPSWithDiscriminator(
            disc_start=disc_start,
            disc_in_channels=disc_in_channels,
            disc_num_layers=disc_num_layers,
            disc_ndf=disc_ndf,
            disc_factor=disc_factor,
            disc_weight=disc_weight,
            disc_loss=disc_loss,
            perceptual_weight=perceptual_weight,
            pixelloss_weight=pixelloss_weight,
            logvar_init=logvar_init,
            lpips_in_channels=lpips_in_channels,
            use_actnorm=use_actnorm,
            use_mask=False,
            use_ivar=False,
        )

        self.disc_lr = disc_lr
        self.ae_lr = ae_lr
        self.ae_betas = tuple(ae_betas)
        self.ae_weight_decay = ae_weight_decay
        self.ae_scheduler_type = ae_scheduler_type
        self.ae_scheduler_t_max = ae_scheduler_t_max
        self.ae_scheduler_eta_min = ae_scheduler_eta_min

        self.val_mse = torchmetrics.MeanSquaredError()

    def get_last_layer(self):
        """Return the last decoder layer weight for adaptive weight calculation."""
        return self.model.upsample.weight

    def loss_fn(self, x):
        """Simplified loss for validation/monitoring (no discriminator)."""
        batch_size = x.shape[0]
        reconstructions = self.model(x)

        # Pixel loss
        diff = x - reconstructions
        pixel_loss = (diff * diff).sum() / batch_size

        # Perceptual loss (only when enabled)
        if self.loss_module.perceptual_weight > 0 and self.loss_module.perceptual_loss is not None:
            inputs_3ch = self.loss_module.channel_adapter(x)
            recon_3ch = self.loss_module.channel_adapter(reconstructions)
            p_loss = self.loss_module.perceptual_loss(inputs_3ch, recon_3ch).mean()
            return pixel_loss + self.loss_module.perceptual_weight * p_loss
        return pixel_loss

    def training_step(self, batch, batch_idx):
        ae_opt, disc_opt = self.optimizers()

        x = batch  # CosmoGrid returns raw tensor
        reconstructions = self.model(x)

        # --- Generator (AE) step ---
        aeloss, log_dict_ae = self.loss_module(
            inputs=x, reconstructions=reconstructions,
            optimizer_idx=0, global_step=self.global_step,
            last_layer=self.get_last_layer(), split="train",
        )

        if torch.isnan(aeloss) or torch.isinf(aeloss):
            self.trainer.should_stop = True
            raise ValueError(f"AE loss is invalid: {aeloss.item()}. Stopping training.")

        ae_opt.zero_grad()
        self.manual_backward(aeloss)
        ae_opt.step()

        # --- Discriminator step ---
        discloss, log_dict_disc = self.loss_module(
            inputs=x, reconstructions=reconstructions,
            optimizer_idx=1, global_step=self.global_step,
            last_layer=self.get_last_layer(), split="train",
        )

        disc_opt.zero_grad()
        self.manual_backward(discloss)
        disc_opt.step()

        # Logging
        self.log("train_ae_loss", aeloss.item(), prog_bar=True)
        self.log("train_disc_loss", discloss.item(), prog_bar=True)
        self.log_dict(log_dict_ae, prog_bar=False, on_step=True, on_epoch=False)
        self.log_dict(log_dict_disc, prog_bar=False, on_step=True, on_epoch=False)

    # noinspection DuplicatedCode
    def validation_step(self, x: "Tensor", batch_idx):
        batch_size = x.shape[0]

        reconstructions = self.model(x)

        # Update MSE metric
        self.val_mse.update(reconstructions.detach().flatten(), x.detach().flatten())
        self.log("val_mse", self.val_mse, batch_size=batch_size, prog_bar=True)

        # Compute losses for logging
        aeloss, log_dict_ae = self.loss_module(
            inputs=x, reconstructions=reconstructions,
            optimizer_idx=0, global_step=self.global_step,
            last_layer=self.get_last_layer(), split="val",
        )
        discloss, log_dict_disc = self.loss_module(
            inputs=x, reconstructions=reconstructions,
            optimizer_idx=1, global_step=self.global_step,
            last_layer=self.get_last_layer(), split="val",
        )

        self.log("val_loss", aeloss.item(), batch_size=batch_size, prog_bar=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

    # noinspection DuplicatedCode
    def configure_optimizers(self):
        ae_params = list(self.model.parameters())
        if self.loss_module.channel_adapter is not None:
            ae_params += list(self.loss_module.channel_adapter.parameters())
        ae_params.append(self.loss_module.logvar)
        ae_opt = optim.AdamW(
            ae_params,
            lr=self.ae_lr,
            betas=self.ae_betas,
            weight_decay=self.ae_weight_decay,
        )

        disc_opt = optim.Adam(
            self.loss_module.discriminator.parameters(),
            lr=self.disc_lr, betas=(0.5, 0.9),
        )

        ae_scheduler_cfg = None
        if self.ae_scheduler_type in ("warmup_cosine", "warmup_cosine_schedule"):
            total_steps = self.trainer.estimated_stepping_batches
            schedule_fn = partial(warmup_cosine_schedule, warmup_steps=self.warmup_steps, total_steps=total_steps)
            ae_scheduler = LambdaLR(ae_opt, lr_lambda=schedule_fn)
            ae_scheduler_cfg = {"scheduler": ae_scheduler, "interval": "step"}
        elif self.ae_scheduler_type in ("cosine_annealing", "cosine"):
            ae_scheduler = CosineAnnealingLR(
                ae_opt,
                T_max=self.ae_scheduler_t_max,
                eta_min=self.ae_scheduler_eta_min,
            )
            ae_scheduler_cfg = {"scheduler": ae_scheduler, "interval": "step"}
        elif self.ae_scheduler_type in ("none", "null", "", None):
            ae_scheduler_cfg = None
        else:
            raise ValueError(f"Unsupported ae_scheduler_type: {self.ae_scheduler_type}")

        return [
            {
                "optimizer": ae_opt,
                **({"lr_scheduler": ae_scheduler_cfg} if ae_scheduler_cfg is not None else {}),
            },
            {"optimizer": disc_opt},
        ]
