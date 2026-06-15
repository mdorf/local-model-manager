# GGUF Introspection + Model Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundation subsystem that scans configured model root directories, parses each `.gguf` file's header, and returns a list of classified `Model` objects — exposed via an `lmm models` CLI.

**Architecture:** Pure-Python, dependency-light core in three focused modules: `gguf.py` (read the GGUF header → metadata + tensor names, never weight data), `models.py` (classify a parsed header into a `Model`), `discovery.py` (recursively scan roots, collapse shards, attach sidecars, degrade gracefully). A thin `cli.py` wires discovery to an `lmm models` command. Tests use a synthetic minimal-GGUF fixture so they never depend on multi-GB files.

**Tech Stack:** Python 3.11+, uv, pytest, ruff, `src/` layout, stdlib `struct`/`mmap`/`argparse` (no third-party runtime deps in this subsystem).

This plan is part of a sequence — see [ROADMAP.md](ROADMAP.md) and [V1_CHECKLIST.md](V1_CHECKLIST.md). It is self-contained and produces a runnable, tested `lmm models`.

---

## File Structure

- `pyproject.toml` — uv project, console script `lmm`, pytest/ruff config.
- `src/lmm/__init__.py` — package marker + version.
- `src/lmm/gguf.py` — `read_gguf(path) -> GGUFInfo`; `GGUFInfo`, `GGUFError`.
- `src/lmm/models.py` — `Model` dataclass; `classify(info, path, shards, sidecars) -> Model`; `quant_from_file_type`.
- `src/lmm/discovery.py` — `discover_models(roots) -> list[Model]`; shard collapsing; sidecar attachment.
- `src/lmm/cli.py` — `main()`; `lmm models` subcommand.
- `tests/conftest.py` — `write_minimal_gguf(...)` helper + fixtures.
- `tests/test_gguf.py`, `tests/test_models.py`, `tests/test_discovery.py`, `tests/test_cli.py`.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `src/lmm/__init__.py`, `tests/test_smoke.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "local-model-manager"
version = "0.0.1"
description = "Manage local llama.cpp model servers and bind agents to them"
requires-python = ">=3.11"
dependencies = []

[project.scripts]
lmm = "lmm.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/lmm"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 2: Create `src/lmm/__init__.py`**

```python
"""local-model-manager core package."""

__version__ = "0.0.1"
```

- [ ] **Step 3: Write a smoke test** — `tests/test_smoke.py`

```python
import lmm


def test_package_imports():
    assert lmm.__version__ == "0.0.1"
```

- [ ] **Step 4: Run it and verify it passes**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS (1 passed). If `uv` is missing, install via `curl -LsSf https://astral.sh/uv/install.sh | sh`.

- [ ] **Step 5: Commit**

```bash
git init -q 2>/dev/null; git add pyproject.toml src/lmm/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold lmm package (uv, pytest)"
```

---

### Task 2: Synthetic GGUF fixture

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the fixture helper** — `tests/conftest.py`

This writes a valid minimal GGUF v3 file (header + typed metadata + tensor-info table; no weight data — matching how `read_gguf` stops before tensor data). Value-type enum per the GGUF spec: UINT32=4, FLOAT32=6, STRING=8, ARRAY=9, UINT64=10.

```python
import struct
from pathlib import Path

import pytest

GGUF_MAGIC = b"GGUF"


def _wstr(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _wval(value) -> bytes:
    # Returns (value_type uint32) + encoded value. Supports int->UINT32,
    # float->FLOAT32, str->STRING, list[str]->ARRAY of STRING.
    if isinstance(value, bool):
        return struct.pack("<I", 7) + struct.pack("<?", value)
    if isinstance(value, int):
        return struct.pack("<I", 4) + struct.pack("<I", value)
    if isinstance(value, float):
        return struct.pack("<I", 6) + struct.pack("<f", value)
    if isinstance(value, str):
        return struct.pack("<I", 8) + _wstr(value)
    if isinstance(value, list):  # array of strings
        out = struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(value))
        for item in value:
            out += _wstr(item)
        return out
    raise TypeError(f"unsupported fixture value: {value!r}")


def write_minimal_gguf(path: Path, metadata: dict, tensor_names: list[str]) -> Path:
    body = bytearray()
    body += GGUF_MAGIC
    body += struct.pack("<I", 3)                      # version
    body += struct.pack("<Q", len(tensor_names))      # tensor_count
    body += struct.pack("<Q", len(metadata))          # metadata_kv_count
    for key, value in metadata.items():
        body += _wstr(key) + _wval(value)
    for name in tensor_names:
        body += _wstr(name)
        body += struct.pack("<I", 1)                  # n_dims = 1
        body += struct.pack("<Q", 1)                  # dims[0] = 1
        body += struct.pack("<I", 0)                  # ggml type = F32
        body += struct.pack("<Q", 0)                  # offset
    path.write_bytes(bytes(body))
    return path


@pytest.fixture
def qwen_like(tmp_path):
    """A Qwen3.6-style hybrid model with an MTP head."""
    meta = {
        "general.architecture": "qwen35",
        "general.name": "Qwen3.6-27B",
        "general.basename": "Qwen3.6-27B",
        "general.size_label": "27B",
        "general.file_type": 7,  # Q8_0
        "general.license.link": "https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/LICENSE",
        "qwen35.block_count": 65,
        "qwen35.context_length": 262144,
        "qwen35.nextn_predict_layers": 1,
        "tokenizer.ggml.tokens": ["<a>", "<b>", "<c>"],  # exercises array skip
    }
    tensors = ["blk.0.attn_q.weight", "blk.64.nextn.eh_proj.weight"]
    return write_minimal_gguf(tmp_path / "Qwen3.6-27B-Q8_0.gguf", meta, tensors)
```

- [ ] **Step 2: Sanity-check the fixture loads as bytes**

Run: `uv run python -c "import struct; from pathlib import Path; import sys; sys.path.insert(0,'tests'); from conftest import write_minimal_gguf; p=write_minimal_gguf(Path('/tmp/x.gguf'), {'general.architecture':'qwen35'}, ['blk.0.attn_q.weight']); print(p.read_bytes()[:4])"`
Expected: prints `b'GGUF'`

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add synthetic minimal-GGUF fixture"
```

---

### Task 3: GGUF header reader

**Files:**
- Create: `src/lmm/gguf.py`
- Test: `tests/test_gguf.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_gguf.py`

```python
import pytest

from lmm.gguf import GGUFError, read_gguf


def test_reads_metadata_scalars_and_strings(qwen_like):
    info = read_gguf(qwen_like)
    assert info.version == 3
    assert info.metadata["general.architecture"] == "qwen35"
    assert info.metadata["general.name"] == "Qwen3.6-27B"
    assert info.metadata["qwen35.block_count"] == 65
    assert info.metadata["qwen35.context_length"] == 262144


def test_reads_tensor_names(qwen_like):
    info = read_gguf(qwen_like)
    assert "blk.0.attn_q.weight" in info.tensor_names
    assert "blk.64.nextn.eh_proj.weight" in info.tensor_names


def test_skips_array_values_without_crashing(qwen_like):
    info = read_gguf(qwen_like)
    # array stored as a marker, not contents
    assert info.metadata["tokenizer.ggml.tokens"]["__array__"] is True
    assert info.metadata["tokenizer.ggml.tokens"]["count"] == 3


def test_rejects_non_gguf(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(GGUFError):
        read_gguf(bad)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_gguf.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lmm.gguf'`

- [ ] **Step 3: Implement `src/lmm/gguf.py`**

```python
"""Read GGUF headers (metadata + tensor names) without loading weight data."""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# GGUF metadata value types -> struct format for scalars.
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I",
           5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_TYPE_STRING = 8
_TYPE_ARRAY = 9


class GGUFError(Exception):
    """Raised when a file is not valid GGUF or cannot be parsed."""


@dataclass
class GGUFInfo:
    version: int
    metadata: dict
    tensor_names: list[str]


class _Reader:
    def __init__(self, mm: mmap.mmap):
        self.mm = mm
        self.off = 0

    def take(self, fmt: str):
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, self.mm, self.off)
        self.off += size
        return vals

    def string(self) -> str:
        (length,) = self.take("<Q")
        s = self.mm[self.off:self.off + length].decode("utf-8", "replace")
        self.off += length
        return s

    def value(self, vtype: int):
        if vtype in _SCALAR:
            return self.take(_SCALAR[vtype])[0]
        if vtype == _TYPE_STRING:
            return self.string()
        if vtype == _TYPE_ARRAY:
            (elem_type,) = self.take("<I")
            (count,) = self.take("<Q")
            if elem_type == _TYPE_STRING:
                for _ in range(count):
                    self.string()
            elif elem_type in _SCALAR:
                self.off += struct.calcsize(_SCALAR[elem_type]) * count
            else:
                raise GGUFError(f"unsupported array element type {elem_type}")
            return {"__array__": True, "elem_type": elem_type, "count": count}
        raise GGUFError(f"unsupported value type {vtype}")


def read_gguf(path: str | Path) -> GGUFInfo:
    path = Path(path)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            r = _Reader(mm)
            (magic,) = r.take("<I")
            if magic != GGUF_MAGIC:
                raise GGUFError(f"{path}: not a GGUF file")
            (version,) = r.take("<I")
            (n_tensors,) = r.take("<Q")
            (n_kv,) = r.take("<Q")
            metadata: dict = {}
            for _ in range(n_kv):
                key = r.string()
                (vtype,) = r.take("<I")
                metadata[key] = r.value(vtype)
            tensor_names: list[str] = []
            for _ in range(n_tensors):
                name = r.string()
                (n_dims,) = r.take("<I")
                r.off += 8 * n_dims        # dims (uint64 each)
                r.take("<I")               # ggml type
                r.take("<Q")               # offset
                tensor_names.append(name)
            return GGUFInfo(version=version, metadata=metadata, tensor_names=tensor_names)
        except struct.error as e:
            raise GGUFError(f"{path}: truncated or malformed GGUF ({e})") from e
        finally:
            mm.close()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_gguf.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lmm/gguf.py tests/test_gguf.py
git commit -m "feat: GGUF header reader (metadata + tensor names)"
```

---

### Task 4: Model classification

**Files:**
- Create: `src/lmm/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_models.py`

```python
from pathlib import Path

from lmm.gguf import read_gguf
from lmm.models import classify, quant_from_file_type


def test_quant_mapping():
    assert quant_from_file_type(7) == "Q8_0"
    assert quant_from_file_type(0) == "F32"
    assert quant_from_file_type(99999) == "unknown"


def test_classify_core_fields(qwen_like):
    m = classify(read_gguf(qwen_like), qwen_like)
    assert m.arch == "qwen35"
    assert m.size_label == "27B"
    assert m.quant == "Q8_0"
    assert m.block_count == 65
    assert m.context_length == 262144
    assert m.family == "qwen3.6"          # basename minus size, lowercased


def test_classify_detects_mtp(qwen_like):
    m = classify(read_gguf(qwen_like), qwen_like)
    assert m.has_mtp is True


def test_classify_extracts_hf_base_repo(qwen_like):
    m = classify(read_gguf(qwen_like), qwen_like)
    assert m.hf_base_repo == "https://huggingface.co/Qwen/Qwen3.6-27B"


def test_classify_passes_through_shards_and_sidecars(qwen_like):
    shards = [qwen_like]
    sidecars = [Path("/models/mmproj.gguf")]
    m = classify(read_gguf(qwen_like), qwen_like, shards=shards, sidecars=sidecars)
    assert m.shards == shards
    assert m.sidecars == sidecars
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lmm.models'`

- [ ] **Step 3: Implement `src/lmm/models.py`**

```python
"""Classify a parsed GGUF header into a Model record."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from lmm.gguf import GGUFInfo

# Partial ggml file_type -> quant label map (extend as needed).
_FILE_TYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_1",
              7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 12: "Q3_K",
              14: "Q4_K", 15: "Q5_K", 16: "Q6_K"}


def quant_from_file_type(file_type: int) -> str:
    return _FILE_TYPE.get(file_type, "unknown")


@dataclass
class Model:
    path: Path
    arch: str
    name: str
    family: str
    size_label: str
    quant: str
    block_count: int | None
    context_length: int | None
    has_mtp: bool
    hf_base_repo: str | None
    shards: list[Path] = field(default_factory=list)
    sidecars: list[Path] = field(default_factory=list)


def _derive_family(basename: str, size_label: str) -> str:
    fam = basename.strip()
    if size_label:
        fam = re.sub(rf"[-_ ]*{re.escape(size_label)}\b", "", fam, flags=re.IGNORECASE)
    return fam.strip("-_ ").lower()


def _hf_repo_from_link(link: str) -> str | None:
    m = re.match(r"(https://huggingface\.co/[^/]+/[^/]+)", link or "")
    return m.group(1) if m else None


def classify(info: GGUFInfo, path: str | Path, *,
             shards: list[Path] | None = None,
             sidecars: list[Path] | None = None) -> Model:
    md = info.metadata
    arch = str(md.get("general.architecture", "unknown"))
    name = str(md.get("general.name", Path(path).stem))
    basename = str(md.get("general.basename", name))
    size_label = str(md.get("general.size_label", ""))
    has_mtp = (
        int(md.get(f"{arch}.nextn_predict_layers", 0) or 0) > 0
        or any(".nextn." in n for n in info.tensor_names)
    )
    block_count = md.get(f"{arch}.block_count")
    context_length = md.get(f"{arch}.context_length")
    return Model(
        path=Path(path),
        arch=arch,
        name=name,
        family=_derive_family(basename, size_label),
        size_label=size_label,
        quant=quant_from_file_type(int(md.get("general.file_type", -1) or -1)),
        block_count=int(block_count) if isinstance(block_count, int) else None,
        context_length=int(context_length) if isinstance(context_length, int) else None,
        has_mtp=has_mtp,
        hf_base_repo=_hf_repo_from_link(str(md.get("general.license.link", ""))),
        shards=shards or [Path(path)],
        sidecars=sidecars or [],
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lmm/models.py tests/test_models.py
git commit -m "feat: classify GGUF metadata into Model records"
```

---

### Task 5: Discovery (scan, shards, sidecars, graceful errors)

**Files:**
- Create: `src/lmm/discovery.py`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_discovery.py`

```python
from lmm.discovery import collapse_shards, discover_models
from tests.conftest import write_minimal_gguf

_META = {"general.architecture": "qwen35", "general.name": "M",
         "general.basename": "M", "general.size_label": "27B",
         "general.file_type": 7, "qwen35.block_count": 4}


def _write(path):
    return write_minimal_gguf(path, _META, ["blk.0.attn_q.weight"])


def test_collapse_shards_groups_by_base():
    names = ["model-00001-of-00003.gguf", "model-00002-of-00003.gguf",
             "model-00003-of-00003.gguf", "solo.gguf"]
    groups = collapse_shards(names)
    assert groups["model.gguf"] == names[:3]
    assert groups["solo.gguf"] == ["solo.gguf"]


def test_discovers_models_recursively(tmp_path):
    _write(tmp_path / "a" / "x.gguf")
    _write(tmp_path / "b" / "c" / "y.gguf")
    models = discover_models([tmp_path])
    assert {m.path.name for m in models} == {"x.gguf", "y.gguf"}


def test_attaches_sidecars(tmp_path):
    d = tmp_path / "m"
    _write(d / "model.gguf")
    (d / "mmproj-model.gguf").write_bytes(b"GGUF")  # sidecar, not a main model
    (d / "template.jinja").write_text("{{x}}")
    models = discover_models([tmp_path])
    main = [m for m in models if m.path.name == "model.gguf"][0]
    names = {p.name for p in main.sidecars}
    assert "mmproj-model.gguf" in names
    assert "template.jinja" in names


def test_skips_unreadable_without_crashing(tmp_path):
    _write(tmp_path / "good.gguf")
    (tmp_path / "broken.gguf").write_bytes(b"NOPE" + b"\x00" * 8)
    models = discover_models([tmp_path])
    assert {m.path.name for m in models} == {"good.gguf"}


def test_missing_root_is_ignored(tmp_path):
    assert discover_models([tmp_path / "does-not-exist"]) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_discovery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lmm.discovery'`

- [ ] **Step 3: Implement `src/lmm/discovery.py`**

```python
"""Recursively discover and classify models under root directories."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from lmm.gguf import GGUFError, read_gguf
from lmm.models import Model, classify

log = logging.getLogger("lmm.discovery")

_SHARD_RE = re.compile(r"^(?P<base>.+)-\d{5}-of-\d{5}\.gguf$")


def collapse_shards(filenames: list[str]) -> dict[str, list[str]]:
    """Group shard filenames under a single logical '<base>.gguf' key.

    Non-sharded files map to themselves. Shard lists are sorted.
    """
    groups: dict[str, list[str]] = {}
    for name in filenames:
        m = _SHARD_RE.match(name)
        key = f"{m.group('base')}.gguf" if m else name
        groups.setdefault(key, []).append(name)
    return {k: sorted(v) for k, v in groups.items()}


def _sidecars(directory: Path, main_names: set[str]) -> list[Path]:
    out: list[Path] = []
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        if p.name in main_names:
            continue
        if p.name.startswith("mmproj") or p.suffix == ".jinja":
            out.append(p)
    return out


def discover_models(roots: list[str | Path]) -> list[Model]:
    models: list[Model] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            log.warning("root not found, skipping: %s", root)
            continue
        by_dir: dict[Path, list[str]] = {}
        for p in root.rglob("*.gguf"):
            by_dir.setdefault(p.parent, []).append(p.name)
        for directory, names in by_dir.items():
            groups = collapse_shards(names)
            main_names = set(groups.keys()) | {n for v in groups.values() for n in v}
            for logical_name, shard_names in groups.items():
                shard_paths = [directory / n for n in shard_names]
                first = shard_paths[0]
                try:
                    info = read_gguf(first)
                except (GGUFError, OSError) as e:
                    log.warning("skipping unreadable model %s: %s", first, e)
                    continue
                models.append(classify(
                    info, first,
                    shards=shard_paths,
                    sidecars=_sidecars(directory, main_names),
                ))
    return models
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_discovery.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lmm/discovery.py tests/test_discovery.py
git commit -m "feat: recursive model discovery with shard + sidecar handling"
```

---

### Task 6: `lmm models` CLI

**Files:**
- Create: `src/lmm/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_cli.py`

```python
from lmm.cli import build_parser, cmd_models
from tests.conftest import write_minimal_gguf

_META = {"general.architecture": "qwen35", "general.name": "Qwen3.6-27B",
         "general.basename": "Qwen3.6-27B", "general.size_label": "27B",
         "general.file_type": 7, "qwen35.block_count": 65,
         "qwen35.nextn_predict_layers": 1}


def test_parser_has_models_subcommand():
    args = build_parser().parse_args(["models", "--root", "/x"])
    assert args.func is cmd_models
    assert args.root == ["/x"]


def test_cmd_models_lists_discovered(tmp_path, capsys):
    write_minimal_gguf(tmp_path / "Qwen3.6-27B-Q8_0.gguf", _META,
                       ["blk.64.nextn.eh_proj.weight"])
    rc = cmd_models(build_parser().parse_args(["models", "--root", str(tmp_path)]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Qwen3.6-27B-Q8_0.gguf" in out
    assert "qwen35" in out
    assert "Q8_0" in out
    assert "MTP" in out  # MTP flag shown for models with a nextn head
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lmm.cli'`

- [ ] **Step 3: Implement `src/lmm/cli.py`**

```python
"""Command-line entrypoint for local-model-manager."""

from __future__ import annotations

import argparse
import logging
import sys

from lmm.discovery import discover_models


def cmd_models(args: argparse.Namespace) -> int:
    models = discover_models(args.root)
    if not models:
        print("No models found.")
        return 0
    for m in sorted(models, key=lambda x: (x.family, x.size_label, x.path.name)):
        flags = " ".join(f for f in ["MTP" if m.has_mtp else ""] if f)
        ctx = f"{m.context_length // 1024}K" if m.context_length else "?"
        print(f"{m.path.name}  [{m.arch} {m.size_label} {m.quant} ctx={ctx}] {flags}".rstrip())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lmm", description="local-model-manager")
    sub = parser.add_subparsers(dest="command", required=True)
    p_models = sub.add_parser("models", help="list discovered local models")
    p_models.add_argument("--root", action="append", default=None,
                          help="model root dir (repeatable); defaults to ~/models")
    p_models.set_defaults(func=cmd_models)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    if getattr(args, "root", None) is None:
        from pathlib import Path
        args.root = [str(Path.home() / "models")]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the whole suite + the real CLI**

Run: `uv run pytest -v && uv run lmm models --root /Users/Shared/models`
Expected: all tests PASS; the CLI prints the three real Qwen3.6-27B models with `[qwen35 27B Q8_0 ctx=256K] MTP` (real-file smoke test — confirms the parser works against actual weights, not just the fixture).

- [ ] **Step 6: Commit**

```bash
git add src/lmm/cli.py tests/test_cli.py
git commit -m "feat: lmm models CLI lists discovered models"
```

---

## Self-Review

**Spec coverage** (against [V1_CHECKLIST.md](V1_CHECKLIST.md) "Model discovery & introspection"):
- Recursive `.gguf` scan ✓ (Task 5) · GGUF header parser w/ arch, blocks, ctx, quant, MTP detect, HF breadcrumb ✓ (Tasks 3–4) · shard collapsing ✓ (Task 5) · sidecars `mmproj`/`.jinja` ✓ (Task 5) · graceful errors ✓ (Task 5) · default `~/models` root ✓ (Task 6).
- **Deferred to later plans (noted, not gaps):** configurable-roots *persistence*, Ollama/LM Studio/HF auto-detect, filesystem watcher, HF mapping override, symlink policy — these belong to the discovery-config / first-run-setup work in plans #4/#6. KV-cache/RAM math and sampler defaults belong to plan #2 (recommendation engine), which consumes this subsystem's output.

**Placeholder scan:** no TBD/TODO/"handle edge cases"; every code step shows full code; every run step shows the exact command + expected output.

**Type consistency:** `read_gguf -> GGUFInfo(version, metadata, tensor_names)` used identically in Tasks 3–6; `classify(info, path, *, shards, sidecars) -> Model` signature consistent in Tasks 4–5; `Model` field names (`has_mtp`, `context_length`, `size_label`, `family`) used consistently in `classify`, discovery, and CLI; `collapse_shards` and `discover_models` names match between impl and tests.
