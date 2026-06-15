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
