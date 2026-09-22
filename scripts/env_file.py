"""Read literal KEY=VALUE settings without executing shell code."""
import re
from pathlib import Path


def read_env_file(path):
    values = {}
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or "\0" in value:
            raise ValueError(f"Invalid env file entry at {path}:{number}; expected KEY=VALUE")
        values[key] = value
    return values
