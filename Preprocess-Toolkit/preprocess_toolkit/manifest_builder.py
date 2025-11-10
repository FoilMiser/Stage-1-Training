"""Utility to print manifest entries from the dataset configuration."""

from __future__ import annotations

import json
import pathlib
from typing import Dict

import yaml


def load_config() -> Dict[str, dict]:
    config_path = pathlib.Path(__file__).resolve().parent / "config" / "datasets.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    for entry in config.values():
        manifest = entry.get("manifest")
        if manifest:
            print(json.dumps(json.loads(manifest), separators=(",", ":")))


if __name__ == "__main__":
    main()
