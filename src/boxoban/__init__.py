from __future__ import annotations

from .env import BoxobanEnv
from .registration import register_envs
from .wrappers import ResizeObservationPIL, TiltedObservationWrapper

register_envs()

__all__ = [
    "BoxobanEnv",
    "ResizeObservationPIL",
    "TiltedObservationWrapper",
    "register_envs",
]
