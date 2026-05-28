"""Simple file-based caching for pipeline stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional


def _file_signature(path: Path) -> str:
    try:
        stat = path.stat()
        return f"{path}:{stat.st_size}:{int(stat.st_mtime)}"
    except FileNotFoundError:
        return f"{path}:missing"


def compute_signature(files: Iterable[Path], extra: Optional[dict] = None) -> str:
    items = [_file_signature(Path(p)) for p in files]
    if extra:
        items.append(json.dumps(extra, sort_keys=True))
    payload = "|".join(items)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path(output_dir: Path) -> Path:
    return output_dir / ".cache.json"


def load_cache(output_dir: Path) -> Optional[dict]:
    path = cache_path(output_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def save_cache(output_dir: Path, signature: str, files: List[Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature,
        "files": [str(p) for p in files],
    }
    cache_path(output_dir).write_text(json.dumps(payload, indent=2))


def should_skip(output_dir: Path, signature: str) -> bool:
    cache = load_cache(output_dir)
    if not cache:
        return False
    return cache.get("signature") == signature
