"""
I/O utility functions.
"""

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path, indent: int = 2):
    """
    Save data to JSON file.

    Args:
        data: Data to save
        path: Output file path
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(path: Path) -> Any:
    """
    Load data from JSON file.

    Args:
        path: Input file path

    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        return json.load(f)
