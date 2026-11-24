# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""OpenVeil: detect, verify and mosaic people in video datasets before release, with an
interactive human review tool.

The core (configuration, box tables, box rules, scan schedule, mask plan and mosaic) needs only
numpy. Video, detection, training and image support are optional extras whose dependencies are
imported only when used.

Frequently used names can be imported from the package directly; they are loaded on first use::

    from openveil import load_config, load_table, MaskPlan
"""
from __future__ import annotations

import importlib
from typing import Any

__version__ = "1.1.0"

_LAZY = {
    "load_config": "config",
    "OpenVeilConfig": "config",
    "BoxTable": "table",
    "load_table": "table",
    "save_table": "table",
    "load_overrides": "table",
    "apply_overrides": "table",
    "giant_rule": "rules",
    "apply_rules_to_table": "rules",
    "firing_episodes": "rules",
    "MaskPlan": "apply.plan",
    "mosaic_array": "apply.mosaic",
    "mosaic_yuv420": "apply.mosaic",
    "block_size": "apply.mosaic",
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'openveil' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY))
