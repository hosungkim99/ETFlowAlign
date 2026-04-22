"""Pocket-aware guidance utilities, including UFF-gradient guidance."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Callable, Optional
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .model import AlignmentBatch

UFFEnergyFn = Callable[[Tensor, AlignmentBatch, "UFFGuidanceConfig"], Tensor]


@dataclass
class UFFGuidanceConfig:
    """Hyperparameters for pocket-aware UFF gradient guidance."""

    scale: float = 1.0
    query_repulsion_weight: float = 0.2
    pocket_attraction_weight: float = 1.0
    pocket_repulsion_weight: float = 0.5
    epsilon: float = 0.1
    sigma: float = 1.5
    backend_mode: str = "auto"  # auto | uff_pytorch | surrogate
    backend_module: str = ""
    backend_symbol: str = ""


class UFFPocketGuidance:
    """Differentiable pocket-aware UFF-style guidance.

    This class computes an energy and returns ``-dE/dx`` for query coordinates.
    If a dedicated UFF backend is available in the runtime, users can swap this
    implementation with that backend while preserving the same call contract.
    """

    def __init__(self, config: Optional[UFFGuidanceConfig] = None, backend_energy_fn: Optional[UFFEnergyFn] = None) -> None:
        self.config = config or UFFGuidanceConfig()
        self._backend_energy_fn = backend_energy_fn or self._resolve_backend_energy_fn()

    def _resolve_backend_energy_fn(self) -> Optional[UFFEnergyFn]:
        """Resolve external UFF backend energy callback if available."""
        mode = self.config.backend_mode
        if mode == "surrogate":
            return None

        if self.config.backend_module:
            return self._load_energy_symbol(self.config.backend_module, self.config.backend_symbol)

        # Best-effort auto integration for UFF_PyTorch-style packages.
        if mode in ("auto", "uff_pytorch"):
            candidates = [
                ("uff_pytorch", "pocket_energy"),
                ("uff_pytorch", "energy"),
                ("UFF_PyTorch", "pocket_energy"),
                ("UFF_PyTorch", "energy"),
            ]
            for module_name, symbol in candidates:
                fn = self._load_energy_symbol(module_name, symbol)
                if fn is not None:
                    return fn
        return None

    def _load_energy_symbol(self, module_name: str, symbol: str) -> Optional[UFFEnergyFn]:
        """Load energy callable from an external module symbol."""
        if not symbol:
            return None
        if importlib.util.find_spec(module_name) is None:
            return None
        module = importlib.import_module(module_name)
        obj = getattr(module, symbol, None)
        if obj is None:
            return None
        if not callable(obj):
            return None
        return obj

    def _lj(self, dist: Tensor, sigma: float, epsilon: float) -> Tensor:
        inv = sigma / dist.clamp_min(1e-6)
        inv6 = inv**6
        inv12 = inv6 * inv6
        return 4.0 * epsilon * (inv12 - inv6)

    def energy(self, x: Tensor, batch: AlignmentBatch) -> Tensor:
        """Compute UFF-like pocket-aware energy for a single alignment state."""
        if self._backend_energy_fn is not None:
            out = self._backend_energy_fn(x, batch, self.config)
            if not torch.is_tensor(out):
                raise TypeError("UFF backend energy must return a torch.Tensor scalar.")
            return out

        e_total = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Query self term: weak steric repulsion to prevent collapse.
        d_qq = torch.cdist(x, x)
        eye = torch.eye(x.size(0), device=x.device, dtype=torch.bool)
        d_qq = d_qq.masked_fill(eye, 1e6)
        e_qq = self._lj(d_qq, sigma=self.config.sigma, epsilon=self.config.epsilon).mean()
        e_total = e_total + self.config.query_repulsion_weight * e_qq

        # Pocket interaction terms.
        if batch.pocket_pos is not None and batch.pocket_pos.numel() > 0:
            d_qp = torch.cdist(x, batch.pocket_pos)
            lj_qp = self._lj(d_qp, sigma=self.config.sigma, epsilon=self.config.epsilon)
            # Balance attraction and repulsion for pocket-aware steering.
            attract = -torch.exp(-d_qp / self.config.sigma).mean()
            repel = lj_qp.clamp_min(0.0).mean()
            e_total = e_total + self.config.pocket_attraction_weight * attract + self.config.pocket_repulsion_weight * repel

        return e_total

    def __call__(self, batch: AlignmentBatch, t_graph: Tensor, v: Tensor) -> Tensor:
        """Return guidance vector field g(x,t) = -∇_x E(x)."""
        del t_graph, v
        x = batch.query_pos
        if not x.requires_grad:
            x = x.detach().clone().requires_grad_(True)
        energy = self.energy(x=x, batch=batch)
        grad = torch.autograd.grad(energy, x, create_graph=False, retain_graph=False)[0]
        return -self.config.scale * grad
