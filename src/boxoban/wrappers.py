"""Observation wrappers for Boxoban environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

__all__ = ["ResizeObservationPIL", "TiltedObservationWrapper"]


class ResizeObservationPIL(gym.ObservationWrapper):
    """Resize image observations using Pillow.

    Lightweight alternative to gymnasium's ``ResizeObservation`` that does not
    require *opencv-python*.

    Parameters
    ----------
    env:
        Environment whose observations are ``HxWxC`` uint8 images.
    shape:
        Target ``(height, width)``.
    resample:
        Resampling filter – ``"nearest"``, ``"bilinear"``, or ``"bicubic"``.
        Default ``"nearest"`` preserves crisp pixel-art edges.
    """

    def __init__(
        self,
        env: gym.Env,
        shape: tuple[int, int],
        resample: str = "nearest",
    ) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        assert len(env.observation_space.shape) in {2, 3}

        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for ResizeObservationPIL. "
                "Install it with: pip install Pillow"
            ) from exc

        _RESAMPLE = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
        }
        if resample not in _RESAMPLE:
            raise ValueError(
                f"resample must be one of {set(_RESAMPLE)}, got {resample!r}"
            )

        self._shape = shape
        self._resample = _RESAMPLE[resample]
        self._Image = Image

        new_shape = shape + env.observation_space.shape[2:]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        img = self._Image.fromarray(observation)
        resized = img.resize(
            (self._shape[1], self._shape[0]),  # PIL uses (width, height)
            resample=self._resample,
        )
        return np.asarray(resized, dtype=np.uint8)


class TiltedObservationWrapper(gym.ObservationWrapper):
    """Apply a perspective transform simulating a tilted camera.

    The top edge of the image is narrowed while the bottom stays unchanged,
    producing a trapezoid that gives a 3-D perspective feel.

    Parameters
    ----------
    env:
        Environment whose observations are ``HxWx3`` uint8 images.
    tilt:
        Fraction of the image width to inset on each side at the top.
        ``0.0`` = identity (no tilt), must be ``< 0.5``.
    fill_color:
        RGB colour for areas outside the trapezoid.
        Defaults to the Boxoban background ``(23, 26, 32)``.
    resample:
        ``"nearest"`` or ``"bilinear"``.  Default ``"bilinear"``.
    """

    def __init__(
        self,
        env: gym.Env,
        tilt: float = 0.2,
        fill_color: tuple[int, int, int] = (23, 26, 32),
        resample: str = "bilinear",
    ) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        assert len(env.observation_space.shape) == 3

        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for TiltedObservationWrapper. "
                "Install it with: pip install Pillow"
            ) from exc

        _RESAMPLE = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
        }
        if resample not in _RESAMPLE:
            raise ValueError(
                f"resample must be one of {set(_RESAMPLE)}, got {resample!r}"
            )

        if not (0.0 <= tilt < 0.5):
            raise ValueError(f"tilt must be in [0.0, 0.5), got {tilt}")

        self._tilt = tilt
        self._fill_color = fill_color
        self._resample = _RESAMPLE[resample]
        self._Image = Image
        self._PERSPECTIVE = Image.Transform.PERSPECTIVE

        H, W = env.observation_space.shape[:2]
        self._img_size = (W, H)  # PIL uses (width, height)
        self._coeffs = self._compute_coeffs(W, H, tilt)

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=env.observation_space.shape,
            dtype=np.uint8,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_coeffs(
        W: int, H: int, tilt: float,
    ) -> tuple[float, ...]:
        """Compute the 8 perspective-transform coefficients.

        Pillow's ``PERSPECTIVE`` uses *inverse* mapping: for each destination
        pixel it looks up the corresponding source pixel.  We solve for the
        coefficients that map **destination-trapezoid → source-rectangle**.

        Source (rectangle)::

            (0,0)-------(W,0)
            |               |
            (0,H)-------(W,H)

        Destination (trapezoid)::

            (inset,0)---(W-inset,0)
               \\             /
            (0,H)-------(W,H)
        """
        inset = W * tilt

        src = [(0, 0), (W, 0), (W, H), (0, H)]
        dst = [(inset, 0), (W - inset, 0), (W, H), (0, H)]

        # Build 8×8 linear system  Ax = b
        matrix: list[list[float]] = []
        rhs: list[float] = []
        for (sx, sy), (dx, dy) in zip(src, dst):
            matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
            matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
            rhs.extend([sx, sy])

        coeffs = np.linalg.solve(
            np.array(matrix, dtype=np.float64),
            np.array(rhs, dtype=np.float64),
        )
        return tuple(coeffs.tolist())

    # ------------------------------------------------------------------
    def observation(self, observation: np.ndarray) -> np.ndarray:
        if self._tilt == 0.0:
            return observation

        img = self._Image.fromarray(observation)
        transformed = img.transform(
            self._img_size,
            self._PERSPECTIVE,
            self._coeffs,
            self._resample,
            fillcolor=self._fill_color,
        )
        return np.asarray(transformed, dtype=np.uint8)
