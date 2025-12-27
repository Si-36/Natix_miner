import torch
import os
from typing import Optional, Dict, Any


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    ema: Optional[Any] = None,
    config: Optional[Dict] = None,
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    if ema is not None:
        checkpoint["ema_shadow"] = ema.shadow

    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[Any] = None,
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if ema is not None and "ema_shadow" in checkpoint:
        ema.shadow = checkpoint["ema_shadow"]

    return checkpoint
