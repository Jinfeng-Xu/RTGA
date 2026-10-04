from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(target)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def summarize_file(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    trace = payload.get("trace", [])
    return {
        "path": str(path),
        "completed": payload.get("completed"),
        "failure_reason": payload.get("failure_reason"),
        "steps_completed": payload.get("steps_completed"),
        "time_completed": payload.get("time_completed"),
        "update_rate": payload.get("update_rate"),
        "guard_rate": payload.get("guard_rate"),
        "max_force_norm": payload.get("max_force_norm"),
        "max_temperature": payload.get("max_temperature"),
        "trace_points": len(trace),
    }


def summarize_cli() -> None:
    parser = argparse.ArgumentParser(description="Summarize RTGA JSON run files")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    for item in args.files:
        print(json.dumps(summarize_file(item), indent=2))

