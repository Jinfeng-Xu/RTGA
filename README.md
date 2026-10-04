# RTGA: Risk-Triggered Guarded Adaptation for MLIP Molecular Dynamics

RTGA is a small controller library for deploying adaptive machine-learning interatomic potentials (MLIPs) in molecular dynamics. It adds calibrated risk-triggered updates and post-force acceptance guards around an existing potential, without prescribing a particular backbone.

The controller is designed for production MD workflows where every accepted force affects the next configuration queried by the model. RTGA separates the deployment loop into three explicit operations:

1. **Risk trigger**: compute a diagnostic score from representation distances, self-supervised losses, uncertainty features, or other model-side signals.
2. **Sparse repair**: apply a local update only when the calibrated risk or a safety schedule requests it.
3. **Guarded acceptance**: check the predicted force and temperature before the force enters the integrator; if unsafe, repair locally and recompute.

## Installation

```bash
git clone <your-rtga-repository-url>
cd RTGA
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development checks:

```bash
pip install -e ".[dev]"
python tests/run_tests.py
```

## Use RTGA With a Formal MLIP

Wrap your MLIP by implementing the `AdaptivePotential` protocol. The wrapper can call any backend: SchNet, PaiNN, NequIP, MACE, Allegro, a custom PyTorch model, or an ASE-compatible calculator.

```python
import numpy as np

from rtga.interfaces import AdaptationInfo, ForcePrediction, MDState


class MyMLIPAdapter:
    def __init__(self, model, optimizer, risk_head):
        self.model = model
        self.optimizer = optimizer
        self.risk_head = risk_head

    def risk_features(self, state: MDState) -> np.ndarray:
        atoms = state.metadata["atoms"]
        return self.risk_head(atoms)

    def evaluate(self, state: MDState) -> ForcePrediction:
        atoms = state.metadata["atoms"]
        energy, forces = self.model.energy_and_forces(atoms)
        return ForcePrediction(energy=float(energy), force=np.asarray(forces))

    def adapt(self, state: MDState, weight: float) -> AdaptationInfo:
        atoms = state.metadata["atoms"]
        loss = self.model.self_supervised_loss(atoms)
        self.optimizer.zero_grad()
        (weight * loss).backward()
        self.optimizer.step()
        return AdaptationInfo(loss=float(loss.detach()), weight=float(weight))
```

Then place the controller around force evaluation:

```python
from rtga.controller import RTGAConfig, RTGAController
from rtga.interfaces import MDState


controller = RTGAController(
    RTGAConfig(
        risk_threshold=2.5,
        risk_slope=1.0,
        periodic_interval=100,
        periodic_weight=0.05,
        force_guard=25.0,
        force_guard_weight=0.5,
        temperature_guard=1200.0,
        temperature_guard_weight=0.5,
        guard_repeats=2,
    )
)

state = MDState(
    position=positions,
    velocity=velocities,
    mass=1.0,
    step=md_step,
    time=md_time_ps,
    metadata={"atoms": atoms, "temperature": current_temperature},
)

prediction = controller.evaluate(adapter, state)
accepted_forces = prediction.force
```

The accepted force returned by `controller.evaluate(...)` is the force that should be passed to the MD integrator.

## Calibrating the Risk Trigger

If your model exposes multi-dimensional diagnostic features, fit a calibration threshold on in-distribution calibration frames:

```python
import numpy as np
from rtga.signals import EmpiricalRiskCalibrator


calibration_features = np.stack([adapter.risk_features(state) for state in calibration_states])
calibrator = EmpiricalRiskCalibrator.fit(calibration_features, quantile=0.90)

score = calibrator.score(adapter.risk_features(state))
```

You can either pass the calibrated score directly through your adapter features or use the calibrator in your MD driver to set `risk_threshold`.

## Running the Included Check

This repository includes a CPU-only executable check that exercises the full controller path and writes a trace file:

```bash
python examples/run_rtga_check.py --config configs/rtga_check.json --output outputs/rtga_check.json
python scripts/summarize_runs.py outputs/rtga_check.json
```

Expected output contains `completed: true`, a non-zero `update_rate`, and a non-zero `guard_rate`.

## Output Trace

`RTGAController.trace_dicts()` returns per-step records containing:

- `step`, `time`
- `risk_score`
- `pre_force_update_weight`
- `guard_update_weight`
- `update_weight`
- `energy`, `force_norm`, `temperature`
- `guard_reason`
- `recomputations`

These fields are intended to support source-data tables, failure-time analysis, and reviewer-auditable MD traces.

## Repository Layout

```text
RTGA/
  configs/              Small runnable configs.
  docs/                 Adapter and release notes.
  examples/             Executable examples.
  scripts/              Result summarization utilities.
  src/rtga/             Installable RTGA package.
  tests/                Lightweight controller checks.
```

## API Surface

- `rtga.interfaces.AdaptivePotential`
- `rtga.interfaces.MDState`
- `rtga.interfaces.ForcePrediction`
- `rtga.interfaces.AdaptationInfo`
- `rtga.controller.RTGAConfig`
- `rtga.controller.RTGAController`
- `rtga.signals.EmpiricalRiskCalibrator`

## Citation

If you use RTGA, please cite the accompanying manuscript and the software metadata in `CITATION.cff`.
