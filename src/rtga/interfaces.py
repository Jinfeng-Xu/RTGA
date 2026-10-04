from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import numpy as np


@dataclass
class MDState:
    """Minimal state object passed between an MD driver and an adaptive potential."""

    position: np.ndarray
    velocity: np.ndarray
    mass: float
    step: int = 0
    time: float = 0.0
    metadata: Dict[str, Any] | None = None

    def kinetic_energy(self) -> float:
        velocity = np.asarray(self.velocity, dtype=float)
        return float(0.5 * self.mass * np.sum(velocity * velocity))


@dataclass
class ForcePrediction:
    """Energy and force predicted by an MLIP-like model."""

    energy: float
    force: np.ndarray
    metadata: Dict[str, Any] | None = None

    @property
    def force_norm(self) -> float:
        return float(np.linalg.norm(np.asarray(self.force, dtype=float)))


@dataclass
class AdaptationInfo:
    """Information returned after a local repair/update."""

    loss: float | None = None
    weight: float = 0.0
    metadata: Dict[str, Any] | None = None


class AdaptivePotential(Protocol):
    """Adapter protocol for an MLIP backbone used by RTGA."""

    def risk_features(self, state: MDState) -> np.ndarray:
        """Return diagnostic features used to compute representation risk."""

    def evaluate(self, state: MDState) -> ForcePrediction:
        """Return energy and force for the current state."""

    def adapt(self, state: MDState, weight: float) -> AdaptationInfo:
        """Apply a local model repair/update at the current state."""

