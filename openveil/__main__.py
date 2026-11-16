# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""``python -m openveil ...`` is the same as the ``openveil`` command."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
