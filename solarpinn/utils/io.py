"""
File I/O utilities.
"""

from __future__ import annotations

from pathlib import Path

def ensure_directory(path: Path) -> None:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path: Path
        Directory path.
    """

    path.mkdir(parents=True, exist_ok=True)

    