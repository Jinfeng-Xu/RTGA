from __future__ import annotations

from dataclasses import dataclass

from rtga.interfaces import ForcePrediction, MDState


@dataclass
class GuardDecision:
    triggered: bool
    weight: float
    reason: str | None = None


@dataclass
class AcceptanceGuard:
    """Post-force safety guard used before accepting forces into the integrator."""

    force_threshold: float | None = None
    force_weight: float = 0.5
    temperature_threshold: float | None = None
    temperature_weight: float = 0.5
    temperature_scale: float = 300.0

    def temperature(self, state: MDState) -> float:
        # The lightweight check uses a single-particle kinetic proxy. Production MD
        # adapters can pass a physically calibrated temperature in metadata.
        if state.metadata and "temperature" in state.metadata:
            return float(state.metadata["temperature"])
        return float(self.temperature_scale * state.kinetic_energy() / max(state.mass, 1e-12))

    def check(self, state: MDState, prediction: ForcePrediction) -> GuardDecision:
        weight = 0.0
        reasons: list[str] = []
        if self.force_threshold is not None and prediction.force_norm >= self.force_threshold:
            weight = max(weight, self.force_weight)
            reasons.append("force")
        temp = self.temperature(state)
        if self.temperature_threshold is not None and temp >= self.temperature_threshold:
            weight = max(weight, self.temperature_weight)
            reasons.append("temperature")
        return GuardDecision(triggered=bool(reasons), weight=weight, reason="+".join(reasons) if reasons else None)
