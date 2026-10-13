"""``python -m laya_pipeline`` entry point."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())