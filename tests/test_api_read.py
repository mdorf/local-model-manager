from fastapi.testclient import TestClient

from lmm.api import create_app
from lmm.daemonconfig import DaemonConfig
from tests.conftest import write_minimal_gguf

_META = {
    "general.architecture": "qwen35", "general.name": "Qwen3.6-27B",
    "general.basename": "Qwen3.6-27B", "general.size_label": "27B",
    "general.file_type": 7, "qwen35.block_count": 65,
    "qwen35.context_length": 262144, "qwen35.full_attention_interval": 4,
    "qwen35.attention.head_count_kv": 4, "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256, "qwen35.nextn_predict_layers": 1,
}
_H = {"Authorization": "Bearer t"}


def _client(root):
    cfg = DaemonConfig(host="127.0.0.1", port=8770, token="t", roots=[str(root)])
    return TestClient(create_app(cfg))


def test_list_models(tmp_path):
    write_minimal_gguf(tmp_path / "Qwen3.6-27B-Q8_0.gguf", _META,
                       ["blk.64.nextn.eh_proj.weight"])
    r = _client(tmp_path).get("/api/models", headers=_H)
    assert r.status_code == 200
    models = r.json()["models"]
    assert len(models) == 1
    assert models[0]["arch"] == "qwen35"
    assert models[0]["has_mtp"] is True


def test_recommend_endpoint(tmp_path):
    write_minimal_gguf(tmp_path / "Qwen3.6-27B-Q8_0.gguf", _META,
                       ["blk.64.nextn.eh_proj.weight"])
    r = _client(tmp_path).get("/api/models/Qwen3.6-27B-Q8_0.gguf/recommend", headers=_H)
    assert r.status_code == 200
    body = r.json()
    assert "flags" in body and "-c" in body["flags"]
    assert body["fit"]["level"] in ("comfortable", "tight", "wont_load")


def test_recommend_unknown_model_404(tmp_path):
    r = _client(tmp_path).get("/api/models/nope.gguf/recommend", headers=_H)
    assert r.status_code == 404
