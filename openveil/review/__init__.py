# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Human review of person boxes: a local web page, its manifest builder, the export and a synthetic demo.

* :mod:`openveil.review.build`  - ``build_from_tables`` / ``build_from_images``: write a review folder.
* :mod:`openveil.review.server` - ``serve`` / ``make_server``: the page on 127.0.0.1 (standard library only).
* :mod:`openveil.review.export` - ``export_review``: human labels, summary, overrides for the apply step.
* :mod:`openveil.review.demo`   - ``make_demo`` / ``run_demo``: synthetic frames for trying the page.

The names below are loaded on first use, so importing this package loads nothing else; frame
extraction (PyAV) and image writing (Pillow or OpenCV) are imported only inside the functions that use them.
"""
from __future__ import annotations

import importlib

_EXPORTS = {
    "build_from_images": "build",
    "build_from_tables": "build",
    "make_demo": "demo",
    "run_demo": "demo",
    "export_review": "export",
    "latest_decisions": "export",
    "DemoGuardError": "server",
    "load_manifest": "server",
    "make_server": "server",
    "read_decisions": "server",
    "serve": "server",
    "validate_manifest": "server",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
