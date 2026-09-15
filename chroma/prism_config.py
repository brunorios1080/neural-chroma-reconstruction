"""Standard-library-only configuration helpers for the Prism experiment family."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path


def merge(base: dict, changes: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_suite(path: str | Path, seen: tuple[Path, ...] = ()) -> dict:
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"Circular Prism configuration inheritance: {path}")
    config = json.loads(path.read_text())
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_suite(path.parent / parent, (*seen, path)), config)
    names = [entry["name"] for entry in config["experiments"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Prism experiment names must be nonempty and unique")
    if any(not re.fullmatch(r"prism_[a-z0-9_]+", name) for name in names):
        raise ValueError("Experiment names must match prism_[a-z0-9_]+")
    return config


def experiment_config(suite: dict, name: str) -> dict:
    matches = [entry for entry in suite["experiments"] if entry["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Unknown Prism experiment: {name}")
    shared = {key: value for key, value in suite.items() if key != "experiments"}
    config = merge(shared, matches[0])
    config["output_dir"] = str(Path(suite["output_dir"]) / name)
    return config
