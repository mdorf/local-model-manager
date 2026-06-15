from lmm.discovery import collapse_shards, discover_models
from tests.conftest import write_minimal_gguf

_META = {"general.architecture": "qwen35", "general.name": "M",
         "general.basename": "M", "general.size_label": "27B",
         "general.file_type": 7, "qwen35.block_count": 4}


def _write(path):
    path.parent.mkdir(parents=True, exist_ok=True)
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
