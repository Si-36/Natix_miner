import torch
import torch.nn as nn
from typing import Optional
import math
from dataclasses import dataclass


@dataclass
class FSAMConfig:
    """F-SAM configuration for modern 2025 implementation."""

    rho: float = 0.5
    eta: float = 0.01
    adaptive: bool = True
    grad_clip: Optional[float] = None


class FSAMOptimizer:
    """
    F-SAM (Sharpness-Aware Minimization) for Phase 4.

    Modern implementation with adaptive rho and gradient clipping.
    Reference: https://arxiv.org/abs/2106.14430 (SAM) + https://arxiv.org/abs/2203.02714 (ASAM)
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        config: Optional[FSAMConfig] = None,
    ):
        self.model = model
        self.base_optimizer = base_optimizer
        self.config = config or FSAMConfig()
        self.state = {}
        self.step_count = 0

    @torch.no_grad()
    def _compute_grad_norm(self) -> float:
        """Compute L2 norm of all gradients."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            total_norm += p.grad.data.norm(2).item() ** 2
        return math.sqrt(total_norm) + 1e-12

    @torch.no_grad()
    def first_step(self, loss: torch.Tensor):
        """First step: maximize loss w.r.t. perturbed parameters."""
        self.base_optimizer.zero_grad()

        # Compute gradient norm
        loss.backward(create_graph=True)
        grad_norm = self._compute_grad_norm()

        # Apply gradient clipping if configured
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            grad_norm = self._compute_grad_norm()

        # Compute epsilon
        epsilon = self.config.rho / grad_norm

        # Perturb parameters in direction of gradient (maximize loss)
        for p in self.model.parameters():
            if p.grad is None:
                continue
            p.data.add_(p.grad.data, alpha=epsilon)

        # Zero gradients for second step
        self.base_optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self, loss: torch.Tensor):
        """Second step: minimize loss at perturbed position."""
        loss.backward()

        # Apply gradient clipping
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

        # Restore original parameters
        epsilon = self.config.rho / self._compute_grad_norm()

        for p in self.model.parameters():
            if p.grad is None:
                continue
            p.data.sub_(p.grad.data, alpha=epsilon)

        # Update with base optimizer
        self.base_optimizer.step()
        self.step_count += 1

    def step(self, closure=None):
        """
        Full F-SAM step.

        Args:
            closure: Callable for computing loss
        """
        loss = closure() if closure is not None else None

        self.first_step(loss)
        loss = closure() if closure is not None else None
        self.second_step(loss)

        return loss

    def state_dict(self) -> dict:
        """Save optimizer state."""
        return {
            "state": self.state,
            "step_count": self.step_count,
            "config": self.config.__dict__,
        }

    def load_state_dict(self, state_dict: dict):
        """Load optimizer state."""
        self.state = state_dict["state"]
        self.step_count = state_dict["step_count"]
        self.config = FSAMConfig(**state_dict["config"])
