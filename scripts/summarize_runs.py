#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from rtga.io import summarize_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RTGA JSON run files")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    for item in args.files:
        print(json.dumps(summarize_file(item), indent=2))


if __name__ == "__main__":
    main()

