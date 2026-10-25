"""Run ``python -m ipd.training`` or ``python ipd/training/__main__.py``."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    for _parent in Path(__file__).resolve().parents:
        if (_parent / "pyproject.toml").is_file():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break

from ipd.training.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
