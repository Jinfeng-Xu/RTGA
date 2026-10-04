from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from rtga.guards import AcceptanceGuard
from rtga.interfaces import AdaptivePotential, ForcePrediction, MDState
from rtga.signals import risk_to_weight


@dataclass
class RTGAConfig:
    risk_threshold: float
    risk_slope: float = 1.0
    periodic_interval: int = 0
    periodic_weight: float = 0.0
    emergency_temperature: float | None = None
    emergency_weight: float = 0.0
    force_guard: float | None = None
    force_guard_weight: float = 0.5
    temperature_guard: float | None = None
    temperature_guard_weight: float = 0.5
    guard_repeats: int = 1
    temperature_scale: float = 300.0


@dataclass
class StepRecord:
    step: int
    time: float
    risk_score: float
    pre_force_update_weight: float
    guard_update_weight: float
    update_weight: float
    energy: float
    force_norm: float
    temperature: float
    guard_reason: str | None
    recomputations: int


class RTGAController:
    """Sparse repair plus post-force guarded acceptance."""

    def __init__(self, config: RTGAConfig):
        self.config = config
        self.guard = AcceptanceGuard(
            force_threshold=config.force_guard,
            force_weight=config.force_guard_weight,
            temperature_threshold=config.temperature_guard,
            temperature_weight=config.temperature_guard_weight,
            temperature_scale=config.temperature_scale,
        )
        self.records: list[StepRecord] = []
        self.total_update_weight = 0.0
        self.guard_update_weight = 0.0
        self.force_evaluations = 0

    def _temperature(self, state: MDState) -> float:
        return self.guard.temperature(state)

    def _pre_force_weight(self, state: MDState, score: float) -> float:
        weight = risk_to_weight(score, self.config.risk_threshold, self.config.risk_slope)
        if self.config.periodic_interval > 0 and state.step > 0 and state.step % self.config.periodic_interval == 0:
            weight = max(weight, self.config.periodic_weight)
        if self.config.emergency_temperature is not None and self._temperature(state) >= self.config.emergency_temperature:
            weight = max(weight, self.config.emergency_weight)
        return float(weight)

    def evaluate(self, model: AdaptivePotential, state: MDState) -> ForcePrediction:
        features = model.risk_features(state)
        score = float(np.linalg.norm(features)) if np.asarray(features).ndim == 0 else float(np.linalg.norm(features))
        pre_weight = self._pre_force_weight(state, score)
        if pre_weight > 0.0:
            model.adapt(state, pre_weight)

        prediction = model.evaluate(state)
        self.force_evaluations += 1
        guard_weight = 0.0
        guard_reason = None
        recomputations = 0

        for _ in range(max(self.config.guard_repeats, 0)):
            decision = self.guard.check(state, prediction)
            if not decision.triggered:
                break
            guard_reason = decision.reason
            model.adapt(state, decision.weight)
            guard_weight += decision.weight
            recomputations += 1
            prediction = model.evaluate(state)
            self.force_evaluations += 1

        total_weight = pre_weight + guard_weight
        self.total_update_weight += total_weight
        self.guard_update_weight += guard_weight
        self.records.append(
            StepRecord(
                step=state.step,
                time=state.time,
                risk_score=score,
                pre_force_update_weight=pre_weight,
                guard_update_weight=guard_weight,
                update_weight=total_weight,
                energy=float(prediction.energy),
                force_norm=prediction.force_norm,
                temperature=self._temperature(state),
                guard_reason=guard_reason,
                recomputations=recomputations,
            )
        )
        return prediction

    @property
    def update_rate(self) -> float:
        return float(self.total_update_weight / max(len(self.records), 1))

    @property
    def guard_rate(self) -> float:
        return float(self.guard_update_weight / max(len(self.records), 1))

    def trace_dicts(self) -> list[dict]:
        return [asdict(record) for record in self.records]

