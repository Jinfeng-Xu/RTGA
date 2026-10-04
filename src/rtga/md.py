from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from rtga.controller import RTGAController
from rtga.interfaces import AdaptivePotential, MDState


@dataclass
class RunSummary:
    completed: bool
    failure_reason: str | None
    steps_completed: int
    time_completed: float
    update_rate: float
    guard_rate: float
    max_force_norm: float
    max_temperature: float
    trace: list[dict]


def velocity_verlet(
    model: AdaptivePotential,
    controller: RTGAController,
    state: MDState,
    steps: int,
    dt: float,
    fail_force: float | None = None,
    fail_temperature: float | None = None,
    temperature_fn: Callable[[MDState], float] | None = None,
) -> RunSummary:
    max_force = 0.0
    max_temperature = 0.0
    completed = True
    failure_reason = None

    prediction = controller.evaluate(model, state)
    force = np.asarray(prediction.force, dtype=float)

    for step in range(1, steps + 1):
        state.velocity = state.velocity + 0.5 * dt * force / state.mass
        state.position = state.position + dt * state.velocity
        state.step = step
        state.time = step * dt
        if temperature_fn is not None:
            state.metadata = dict(state.metadata or {})
            state.metadata["temperature"] = temperature_fn(state)

        prediction = controller.evaluate(model, state)
        force = np.asarray(prediction.force, dtype=float)
        state.velocity = state.velocity + 0.5 * dt * force / state.mass

        temp = controller.records[-1].temperature
        max_force = max(max_force, controller.records[-1].force_norm)
        max_temperature = max(max_temperature, temp)

        if not np.isfinite(prediction.energy) or not np.all(np.isfinite(force)):
            completed = False
            failure_reason = "nonfinite_prediction"
            break
        if fail_force is not None and controller.records[-1].force_norm > fail_force:
            completed = False
            failure_reason = "force_failure"
            break
        if fail_temperature is not None and temp > fail_temperature:
            completed = False
            failure_reason = "temperature_failure"
            break

    return RunSummary(
        completed=completed,
        failure_reason=failure_reason,
        steps_completed=state.step,
        time_completed=state.time,
        update_rate=controller.update_rate,
        guard_rate=controller.guard_rate,
        max_force_norm=max_force,
        max_temperature=max_temperature,
        trace=controller.trace_dicts(),
    )


def summary_to_dict(summary: RunSummary) -> dict:
    return asdict(summary)

