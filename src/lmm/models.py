"""Classify a parsed GGUF header into a Model record."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from lmm.gguf import GGUFInfo

# Partial ggml file_type -> quant label map (extend as needed).
_FILE_TYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
              7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 12: "Q3_K",
              14: "Q4_K", 15: "Q5_K", 16: "Q6_K"}


def quant_from_file_type(file_type: int) -> str:
    return _FILE_TYPE.get(file_type, "unknown")


def _as_int(value: object, default: int | None = None) -> int | None:
    """Return value as int when it's a real int, else default (arrays/markers/None)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else default


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
    license: str | None = None
    quantized_by: str | None = None
    has_chat_template: bool = False


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
        (_as_int(md.get(f"{arch}.nextn_predict_layers"), 0) or 0) > 0
        or any(".nextn." in n for n in info.tensor_names)
    )
    block_count = md.get(f"{arch}.block_count")
    context_length = md.get(f"{arch}.context_length")
    # HF model-card link: prefer the license link, fall back to the base-model /
    # general repo_url that many quants carry.
    hf_repo = (_hf_repo_from_link(str(md.get("general.license.link", "")))
               or _hf_repo_from_link(str(md.get("general.base_model.0.repo_url", "")))
               or _hf_repo_from_link(str(md.get("general.repo_url", ""))))
    return Model(
        path=Path(path),
        arch=arch,
        name=name,
        family=_derive_family(basename, size_label),
        size_label=size_label,
        quant=quant_from_file_type(_as_int(md.get("general.file_type"), -1)),
        block_count=_as_int(block_count),
        context_length=_as_int(context_length),
        has_mtp=has_mtp,
        hf_base_repo=hf_repo,
        shards=shards or [Path(path)],
        sidecars=sidecars or [],
        license=(str(md["general.license"]) if md.get("general.license") else None),
        quantized_by=(str(md["general.quantized_by"]) if md.get("general.quantized_by") else None),
        has_chat_template=bool(md.get("tokenizer.chat_template")),
    )
