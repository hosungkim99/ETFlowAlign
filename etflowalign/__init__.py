"""Public API for ETFlowAlign.

This module re-exports the main classes/functions so users can import from
``etflowalign`` directly without navigating submodules.
"""

""Public API for the ETFlowAlign scaffold."""

from .flow_matching import AlignmentFlowMatcher, FlowMatchingConfig, flow_matching_step
from .inference import generate_candidates, rank_candidates
from .model import AlignmentBatch, ETFlowAlignModel
from .sampler import ETFlowAlignSampler, ODESamplerConfig
from .train import TrainConfig, build_training_components, train_step

__all__ = [
    "AlignmentBatch",
    "AlignmentFlowMatcher",
    "ETFlowAlignModel",
    "ETFlowAlignSampler",
    "FlowMatchingConfig",
    "ODESamplerConfig",
    "TrainConfig",
    "build_training_components",
    "flow_matching_step",
    "generate_candidates",
    "rank_candidates",
    "train_step",
]
