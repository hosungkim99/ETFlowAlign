"""flow-matching 코어: prior(시작분포) · path(보간경로) · loss(목적함수).

백본이 예측하는 속도장 v_θ 를 학습/샘플링에 연결하는 프리미티브.
SPEC.md §4 참조.
"""

from etflowalign.flow.loss import batchwise_l2_loss
from etflowalign.flow.path import (
    center_pos,
    interpolate,
    kabsch_aligned_rmsd,
    sample_time,
    target_velocity,
)
from etflowalign.flow.prior import GaussianSampler, HarmonicSampler, get_prior

__all__ = [
    "batchwise_l2_loss",
    "center_pos",
    "interpolate",
    "kabsch_aligned_rmsd",
    "sample_time",
    "target_velocity",
    "GaussianSampler",
    "HarmonicSampler",
    "get_prior",
]
