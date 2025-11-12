"""辞書ファイルを読み込むためのユーティリティ。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable


DICTIONARY_DIR = Path(__file__).parent


@lru_cache()
def load_list(filename: str) -> list[str]:
    path = DICTIONARY_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"dictionary file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"dictionary file must contain a JSON list: {path}")
    return data


@lru_cache()
def load_set(filename: str) -> set[str]:
    return set(load_list(filename))


def ensure_strings(values: Iterable[str]) -> set[str]:
    return {str(value) for value in values}

