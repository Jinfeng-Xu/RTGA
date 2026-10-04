from __future__ import annotations

import numpy as np

from rtga.controller import RTGAConfig, RTGAController
from rtga.interfaces import MDState
from rtga.check_potential import CheckPotential


def test_guard_triggers_recompute_for_large_force() -> None:
    state = MDState(position=np.array([0.82]), velocity=np.array([0.0]), mass=1.0)
    model = CheckPotential(spike_strength=6.0)
    controller = RTGAController(
        RTGAConfig(
            risk_threshold=99.0,
            force_guard=1.0,
            force_guard_weight=0.8,
            guard_repeats=2,
        )
    )
    prediction = controller.evaluate(model, state)
    assert prediction.force_norm < 6.0
    assert controller.records[-1].recomputations >= 1
    assert controller.guard_rate > 0.0


def test_risk_trigger_updates_without_guard() -> None:
    state = MDState(position=np.array([1.25]), velocity=np.array([0.0]), mass=1.0)
    model = CheckPotential()
    controller = RTGAController(
        RTGAConfig(
            risk_threshold=0.1,
            risk_slope=1.0,
            force_guard=100.0,
            guard_repeats=1,
        )
    )
    controller.evaluate(model, state)
    assert controller.records[-1].pre_force_update_weight > 0.0
    assert controller.records[-1].guard_update_weight == 0.0
    assert controller.update_rate > 0.0
