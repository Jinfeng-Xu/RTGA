#!/usr/bin/env python
from __future__ import annotations

from test_controller import test_guard_triggers_recompute_for_large_force, test_risk_trigger_updates_without_guard


def main() -> None:
    test_guard_triggers_recompute_for_large_force()
    test_risk_trigger_updates_without_guard()
    print("RTGA lightweight tests passed")


if __name__ == "__main__":
    main()

