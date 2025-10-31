from __future__ import annotations

"""Deprecated thin wrapper. Delegates to src.main to keep a single entrypoint.

Kept for backward compatibility with earlier instructions/README.
"""

from typing import Optional, List

from src.main import main as _main


def main(argv: Optional[List[str]] = None) -> int:
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
