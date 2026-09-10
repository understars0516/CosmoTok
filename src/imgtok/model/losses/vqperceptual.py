import torch
import torch.nn.functional as F


def adopt_weight(weight: float, global_step: int, threshold: int = 0) -> float:
    """Zero out weight until global_step reaches threshold (discriminator warmup)."""
    if threshold > global_step:
        return 0.0
    return weight


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator."""
    loss_real = F.relu(1.0 - logits_real).mean()
    loss_fake = F.relu(1.0 + logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Vanilla binary cross-entropy loss for discriminator."""
    loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
    return 0.5 * (loss_real + loss_fake)
