"""Pure helpers for reading/streaming llama-server log files."""
from __future__ import annotations
from pathlib import Path


def read_log_tail(path: str | Path, max_lines: int = 200) -> list[str]:
    try:
        data = Path(path).read_text(errors="replace")
    except OSError:
        return []
    lines = data.splitlines()
    return lines[-max_lines:]


def tail_new_lines(path: str | Path, offset: int) -> tuple[list[str], int]:
    """Return (complete new lines since `offset`, new byte offset)."""
    p = Path(path)
    try:
        with p.open("rb") as f:
            f.seek(offset)
            chunk = f.read()
    except OSError:
        return [], offset
    if not chunk:
        return [], offset
    text = chunk.decode("utf-8", errors="replace")
    # only emit complete lines; keep the offset at the last newline boundary
    last_nl = text.rfind("\n")
    if last_nl == -1:
        return [], offset
    complete = text[: last_nl + 1]
    new_offset = offset + len(complete.encode("utf-8"))
    return complete.splitlines(), new_offset
