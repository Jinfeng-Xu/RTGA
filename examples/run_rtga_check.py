#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rtga.controller import RTGAConfig, RTGAController
from rtga.interfaces import MDState
from rtga.io import write_json
from rtga.md import summary_to_dict, velocity_verlet
from rtga.check_potential import CheckPotential


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight RTGA controller check")
    parser.add_argument("--config", default="configs/rtga_check.json")
    parser.add_argument("--output", default="outputs/rtga_check.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    state = MDState(
        position=np.array([float(cfg["initial_position"])], dtype=float),
        velocity=np.array([float(cfg["initial_velocity"]) + 0.0 * rng.normal()], dtype=float),
        mass=float(cfg["mass"]),
        metadata={"temperature": 0.0},
    )
    model = CheckPotential()
    controller = RTGAController(
        RTGAConfig(
            risk_threshold=float(cfg["risk_threshold"]),
            risk_slope=float(cfg.get("risk_slope", 1.0)),
            periodic_interval=int(cfg.get("periodic_interval", 0)),
            periodic_weight=float(cfg.get("periodic_weight", 0.0)),
            force_guard=float(cfg["force_guard"]),
            force_guard_weight=float(cfg["force_guard_weight"]),
            temperature_guard=float(cfg["temperature_guard"]),
            temperature_guard_weight=float(cfg["temperature_guard_weight"]),
            guard_repeats=int(cfg.get("guard_repeats", 1)),
            temperature_scale=float(cfg.get("temperature_scale", 300.0)),
        )
    )

    def temperature(current: MDState) -> float:
        return float(cfg.get("temperature_scale", 300.0) * current.kinetic_energy() / max(current.mass, 1e-12))

    summary = velocity_verlet(
        model=model,
        controller=controller,
        state=state,
        steps=int(cfg["steps"]),
        dt=float(cfg["dt"]),
        fail_force=float(cfg["failure_force"]),
        fail_temperature=float(cfg["failure_temperature"]),
        temperature_fn=temperature,
    )
    payload = summary_to_dict(summary)
    payload["config"] = cfg
    payload["final_model"] = {
        "k_model": model.k_model,
        "spike_strength": model.spike_strength,
    }
    write_json(args.output, payload)
    print(json.dumps({k: payload[k] for k in ["completed", "failure_reason", "steps_completed", "update_rate", "guard_rate"]}, indent=2))


if __name__ == "__main__":
    main()
