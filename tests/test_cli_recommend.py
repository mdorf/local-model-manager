from lmm.cli import build_parser, cmd_recommend
from tests.conftest import write_minimal_gguf

_META = {
    "general.architecture": "qwen35", "general.name": "Qwen3.6-27B",
    "general.basename": "Qwen3.6-27B", "general.size_label": "27B",
    "general.file_type": 7, "qwen35.block_count": 65,
    "qwen35.context_length": 262144, "qwen35.full_attention_interval": 4,
    "qwen35.attention.head_count_kv": 4, "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256, "qwen35.nextn_predict_layers": 1,
}


def test_parser_has_recommend_subcommand():
    args = build_parser().parse_args(["recommend", "m.gguf", "--root", "/x"])
    assert args.func is cmd_recommend
    assert args.model == "m.gguf"
    assert args.root == ["/x"]


def test_cmd_recommend_prints_config(tmp_path, capsys):
    write_minimal_gguf(tmp_path / "Qwen3.6-27B-Q8_0.gguf", _META,
                       ["blk.64.nextn.eh_proj.weight"])
    rc = cmd_recommend(build_parser().parse_args(
        ["recommend", "Qwen3.6-27B-Q8_0.gguf", "--root", str(tmp_path)]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "llama-server" in out
    assert "-c" in out
    assert "draft-mtp" in out
    assert any(w in out.lower() for w in ("comfortable", "tight", "won't", "wont"))


def test_cmd_recommend_unknown_model_returns_error(tmp_path, capsys):
    rc = cmd_recommend(build_parser().parse_args(
        ["recommend", "missing.gguf", "--root", str(tmp_path)]))
    assert rc == 1
    assert "not found" in capsys.readouterr().out.lower()
