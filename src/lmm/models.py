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
