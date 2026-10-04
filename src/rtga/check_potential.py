from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rtga.interfaces import AdaptationInfo, ForcePrediction, MDState


@dataclass
class CheckPotential:
    """Deterministic potential used by the lightweight RTGA controller check."""

    k_true: float = 1.0
    k_model: float = 0.55
    spike_center: float = 0.82
    spike_width: float = 0.10
    spike_strength: float = 3.5
    repair_rate: float = 0.22

    def risk_features(self, state: MDState) -> np.ndarray:
        x = float(np.asarray(state.position).reshape(-1)[0])
        distance_to_spike = abs(x - self.spike_center)
        shift = max(abs(x) - 0.65, 0.0)
        local_spike = np.exp(-(distance_to_spike**2) / max(2 * self.spike_width**2, 1e-12))
        return np.array([shift, local_spike], dtype=float)

    def evaluate(self, state: MDState) -> ForcePrediction:
        x = float(np.asarray(state.position).reshape(-1)[0])
        spike = self.spike_strength * np.exp(-((x - self.spike_center) ** 2) / max(2 * self.spike_width**2, 1e-12))
        force_value = -self.k_model * x + spike
        energy = 0.5 * self.k_model * x * x + self.spike_strength * self.spike_width * np.sqrt(2 * np.pi)
        return ForcePrediction(energy=float(energy), force=np.array([force_value], dtype=float))

    def adapt(self, state: MDState, weight: float) -> AdaptationInfo:
        weight = float(np.clip(weight, 0.0, 1.0))
        before_k = self.k_model
        before_spike = self.spike_strength
        self.k_model += self.repair_rate * weight * (self.k_true - self.k_model)
        self.spike_strength *= max(0.0, 1.0 - 0.35 * weight)
        loss = abs(self.k_true - self.k_model) + abs(self.spike_strength)
        return AdaptationInfo(
            loss=float(loss),
            weight=weight,
            metadata={
                "k_model_before": before_k,
                "k_model_after": self.k_model,
                "spike_before": before_spike,
                "spike_after": self.spike_strength,
            },
        )

