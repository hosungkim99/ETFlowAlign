"""Minimal training entry utilities for ETFlowAlign.

This is intentionally small: it wires model + matcher without enforcing
any particular trainer framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import optim

from .flow_matching import AlignmentFlowMatcher, FlowMatchingConfig, flow_matching_step
from .model import AlignmentBatch, ETFlowAlignModel


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0


def build_training_components(
    model: ETFlowAlignModel,
    train_config: TrainConfig,
    fm_config: FlowMatchingConfig,
):
    matcher = AlignmentFlowMatcher(config=fm_config)
    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    return matcher, optimizer


def train_step(
    model: ETFlowAlignModel,
    matcher: AlignmentFlowMatcher,
    optimizer: optim.Optimizer,
    batch: AlignmentBatch,
    target_query_pos: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = flow_matching_step(model=model, matcher=matcher, batch=batch, target_query_pos=target_query_pos)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())
