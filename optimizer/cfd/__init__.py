"""CFD layer: mesh requirements, SimScale backend, mock backend, result parsing."""
from __future__ import annotations

from ..config import Config


def make_backend(name: str, cfg: Config):
    """Backend factory: 'simscale' (real CFD) or 'mock' (analytic surrogate for dry runs)."""
    if name == "mock":
        from .mock_backend import MockBackend

        return MockBackend()
    if name == "simscale":
        from .simscale_backend import SimScaleBackend

        return SimScaleBackend(cfg)
    raise ValueError(f"unknown CFD backend '{name}' (use 'simscale' or 'mock')")
